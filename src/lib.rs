use std::collections::HashMap;
use std::path::Path;

mod error;
mod parse;
pub use error::Error;

#[derive(Debug)]
pub struct Hdf5File {
    map: memmap::Mmap,
    root_attributes: HashMap<String, Attribute>,
    root_datasets: HashMap<String, Dataset>,
    groups: HashMap<String, Group>,
}

impl Hdf5File {
    pub fn view<T: Hdf5Type>(&self, name: &str) -> &[T] {
        let dataset = self.root_datasets.get(name).unwrap();
        if T::dtype() != dataset.dtype {
            panic!("invalid type passed to view");
        }
        unsafe {
            use std::ops::Deref;
            let data_start = self.map.deref().as_ptr().offset(dataset.address as isize) as *const T;
            std::slice::from_raw_parts(data_start, dataset.size as usize / std::mem::size_of::<T>())
        }
    }
}

#[derive(Clone, Debug)]
pub struct Group {
    name: String,
    datasets: HashMap<String, Dataset>,
    attributes: HashMap<String, Attribute>,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    dimensions: Vec<u64>,
    dtype: Hdf5Dtype,
    address: u64,
    size: u64,
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

        let root_node = parse::hdf5_node(
            &contents[superblock.root_group_symbol_table_entry.address_of_btree as usize..],
            superblock.offset_size,
        )?
        .1;

        let name_heap = parse::parse_local_heap(
            &contents[superblock
                .root_group_symbol_table_entry
                .address_of_name_heap as usize..],
            superblock.offset_size,
            superblock.length_size,
        )?
        .1;

        let root_groups = if let parse::BtreeNode::GroupNode { entries, .. } = root_node {
            entries
        } else {
            unreachable!("Root node ought to be a group node");
        };

        let mut root_datasets = HashMap::new();
        let groups = HashMap::new();

        for root_group in root_groups {
            let table = parse::parse_symbol_table(
                &contents[root_group.pointer_to_symbol_table as usize..],
                superblock.offset_size,
            )?
            .1;

            for object in &table.entries {
                let name = &contents
                    [(object.link_name_offset + name_heap.address_of_data_segment) as usize..]
                    .iter()
                    .take_while(|b| **b != 0)
                    .map(|b| *b as char)
                    .collect::<String>();

                let object_header = parse::parse_object_header(
                    &contents[object.object_header_address as usize..],
                    6,
                )?
                .1;

                // Datasets have a data layout, data storage, datatype, and dataspace
                let mut layout = None;
                let mut fillvalue = None;
                let mut dtype = None;
                let mut dspace = None;
                let mut attributes = Vec::new();
                let mut symbol_table = None;

                use parse::header::Message;
                for message in object_header.messages {
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
                        root_datasets.insert(name.clone(), Dataset::from(dspace, dtype, layout));
                    }
                    (None, None, None, None, Some(_symbol_table)) => {
                        println!("found group with name {:?}", name);
                    }
                    _ => unreachable!("Unrecognized kind of HDF5 object found"),
                }
            }
        }

        Ok(Hdf5File {
            map: contents,
            groups,
            root_datasets,
            root_attributes: HashMap::new(),
        })
    }
}

#[test]
fn open_file() {
    let file = Hdf5File::open("test.hdf5").unwrap();
    println!("{:#?}", file);
    assert!(file.groups.len() == 1);
    assert!(file.groups.contains_key("test_dataset"));
}
