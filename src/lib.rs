use std::collections::BTreeMap; // Currently use BTreeMap just to get sorted Debug output
use std::path::Path;

mod error;
mod parse;
pub use error::Error;

pub fn open<P: AsRef<Path>>(path: P) -> Result<Hdf5File, Error> {
    Hdf5File::open(path)
}

pub struct Hdf5File {
    map: memmap::Mmap,
    root_attributes: BTreeMap<String, Attribute>,
    root_datasets: BTreeMap<String, Dataset>,
    groups: BTreeMap<String, Group>,
}

impl std::fmt::Debug for Hdf5File {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Hdf5File")
            .field("attributes", &self.root_attributes)
            .field("datasets", &self.root_datasets)
            .field("groups", &self.groups)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct Group {
    attributes: BTreeMap<String, Attribute>,
    datasets: BTreeMap<String, Dataset>,
    groups: BTreeMap<String, Group>,
}

#[derive(Clone)]
pub struct Dataset {
    dimensions: Vec<u64>,
    dtype: Hdf5Dtype,
    address: u64,
    size: u64,
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("dimensions", &self.dimensions)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl Dataset {
    fn from(
        dataspace: parse::header::Dataspace,
        datatype: parse::header::DataType,
        layout: parse::header::DataLayout,
    ) -> Self {
        Dataset {
            dimensions: dataspace.dimensions,
            dtype: Hdf5Dtype::from(datatype),
            address: layout.address,
            size: layout.size,
        }
    }
}

pub trait Hdf5Type {
    fn dtype() -> Hdf5Dtype;
}

impl Hdf5Type for f64 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::F64
    }
}

impl Hdf5Type for f32 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::F32
    }
}

impl Hdf5Type for i64 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::I64
    }
}

impl Hdf5Type for i32 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::I32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hdf5Dtype {
    F64,
    F32,
    I64,
    I32,
}

impl Hdf5Dtype {
    fn from(raw: parse::header::DataType) -> Self {
        match (raw.class, raw.size) {
            (0, 8) => Hdf5Dtype::I64,
            (0, 4) => Hdf5Dtype::I32,
            (1, 8) => Hdf5Dtype::F64,
            (1, 4) => Hdf5Dtype::F32,
            _ => unimplemented!("dtype not supported yet {:#?}", raw),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Attribute {
    name: String,
}

impl Hdf5File {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = std::fs::File::open(path)?;
        let contents = unsafe { memmap::Mmap::map(&file)? };
        let superblock = parse::parse_superblock(&contents)?.1;

        use std::ops::Deref;
        let root = parse_group(
            contents.deref(),
            parse::header::SymbolTable {
                btree_address: superblock.root_group_symbol_table_entry.address_of_btree,
                local_heap_address: superblock
                    .root_group_symbol_table_entry
                    .address_of_name_heap,
            },
        )?;

        Ok(Hdf5File {
            map: contents,
            groups: root.groups,
            root_datasets: root.datasets,
            root_attributes: root.attributes,
        })
    }

    pub fn view<T: Hdf5Type>(&self, name: &str) -> &[T] {
        let dataset = self.root_datasets.get(name).unwrap();
        if T::dtype() != dataset.dtype {
            eprintln!(
                "Dataset {:?} is of type {:?}, not {:?}",
                name,
                dataset.dtype,
                T::dtype()
            );
            panic!("invalid type passed to view");
        }
        unsafe {
            use std::ops::Deref;
            let data_start = self.map.deref().as_ptr().offset(dataset.address as isize) as *const T;
            std::slice::from_raw_parts(data_start, dataset.size as usize / std::mem::size_of::<T>())
        }
    }
}

fn parse_group(contents: &[u8], symbol_table: parse::header::SymbolTable) -> Result<Group, Error> {
    let node = parse::hdf5_node(&contents[symbol_table.btree_address as usize..], 8)?.1;

    let name_heap =
        parse::parse_local_heap(&contents[symbol_table.local_heap_address as usize..], 8, 8)?.1;

    let mut datasets = BTreeMap::new();
    let mut groups = BTreeMap::new();
    let attributes = BTreeMap::new();

    for group_entry in node.entries {
        let table = parse::parse_symbol_table(
            &contents[group_entry.pointer_to_symbol_table as usize..],
            8,
        )?
        .1;

        for object in &table.entries {
            let name = &contents
                [(object.link_name_offset + name_heap.address_of_data_segment) as usize..]
                .iter()
                .take_while(|b| **b != 0)
                .map(|b| *b as char)
                .collect::<String>();

            use parse::header::Message;
            use parse::header::ObjectHeaderContinuation;
            let (mut remaining, object_header) =
                parse::parse_object_header(&contents[object.object_header_address as usize..])?;
            let mut messages = Vec::new();

            for _ in 0..object_header.total_number_of_header_messages {
                let (rem, message) = parse::parse_header_message(remaining)?;
                if let Message::ObjectHeaderContinuation(ObjectHeaderContinuation {
                    offset, ..
                }) = message
                {
                    remaining = &contents[offset as usize..];
                } else {
                    messages.push(message.clone());
                    remaining = rem;
                }
                // Attribute parsing is broken, which throws off the parser as soon as it hits
                // one
                if let Message::Attribute(_) = message {
                    break;
                }
            }

            // Datasets have a data layout, data storage, datatype, and dataspace
            let mut layout = None;
            let mut fillvalue = None;
            let mut dtype = None;
            let mut dspace = None;
            let mut attributes = Vec::new();
            let mut symbol_table = None;

            for message in messages {
                match message {
                    Message::DataLayout(m) => layout = Some(m),
                    Message::DataStorageFillValue(m) => fillvalue = Some(m),
                    Message::DataType(m) => dtype = Some(m),
                    Message::Dataspace(m) => dspace = Some(m),
                    Message::Attribute(m) => attributes.push(m),
                    Message::SymbolTable(m) => symbol_table = Some(m),
                    _ => {}
                }
            }

            match (layout, fillvalue, dtype, dspace, symbol_table) {
                (Some(layout), Some(_fillvalue), Some(dtype), Some(dspace), None) => {
                    datasets.insert(name.clone(), Dataset::from(dspace, dtype, layout));
                }
                (None, None, None, None, Some(symbol_table)) => {
                    groups.insert(name.clone(), parse_group(contents, symbol_table)?);
                }
                _ => eprintln!("Unrecognized kind of HDF5 object found"),
            }
        }
    }

    Ok(Group {
        datasets,
        attributes,
        groups,
    })
}
