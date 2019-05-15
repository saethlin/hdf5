//! A pure-Rust HDF5 library, built for speed
//!
//! This library does not intend to support all features of HDF5 either in the library or the
//! specification.

use std::collections::BTreeMap; // Currently use BTreeMap just to get sorted Debug output
use std::path::Path;

mod error;
mod parse;
pub use error::Error;

/// Convienence function for Hdf5File::open
pub fn open<P: AsRef<Path>>(path: P) -> Result<Hdf5File, Error> {
    Hdf5File::open(path)
}

/// An opened HDF5 file
pub struct Hdf5File {
    map: memmap::Mmap,
    root_group: Group,
}

impl std::fmt::Debug for Hdf5File {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Hdf5File")
            .field("attributes", &self.root_group.attributes)
            .field("datasets", &self.root_group.datasets)
            .field("groups", &self.root_group.groups)
            .finish()
    }
}

#[derive(Debug)]
struct Group {
    attributes: BTreeMap<String, Attribute>,
    datasets: BTreeMap<String, Dataset>,
    groups: BTreeMap<String, Group>,
}

impl Group {
    fn find_dataset(&self, dataset_path: &str) -> &Dataset {
        let delim_index = dataset_path.find("/");
        if let Some(i) = delim_index {
            let (first, remaining) = dataset_path.split_at(i);
            if let Some(d) = self.datasets.get(first) {
                d
            } else {
                self.groups[first].find_dataset(&remaining[1..])
            }
        } else {
            &self.datasets[dataset_path]
        }
    }
}

struct Dataset {
    dimensions: Vec<u64>,
    dtype: Hdf5Dtype,
    address: u64,
    size: u64,
}

#[derive(Debug)]
struct Attribute {}

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

/// Identifies Rust types that this library can produce from HDF5 types
pub trait Hdf5Type: private::Sealed {
    fn dtype() -> Hdf5Dtype;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for i64 {}
    impl Sealed for i32 {}
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

#[doc(hidden)]
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

impl Hdf5File {
    /// Open an HDF5 file
    ///
    /// This function memory-maps the file and initializes a number of internal data structures to
    /// make access to data trivial.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = std::fs::File::open(path)?;
        let contents = unsafe { memmap::Mmap::map(&file)? };
        let superblock = parse::parse_superblock(&contents)?.1;

        use std::ops::Deref;
        let root_group = parse_group(
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
            root_group,
        })
    }

    /// Look up the provided path to a dataset, if one is found and its type correct, return a
    /// slice of the underlying file mapping.
    ///
    /// Note that this discards any dimension information associated with the dataset.
    ///
    /// Panics if the given path doesn't lead to a datset or the type is incorrect.
    pub fn view<T: Hdf5Type>(&self, dataset_path: &str) -> &[T] {
        let dataset = self.root_group.find_dataset(dataset_path);
        if T::dtype() != dataset.dtype {
            panic!(
                "Dataset {:?} is of type {:?}, not {:?}",
                dataset_path,
                dataset.dtype,
                T::dtype()
            );
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
