//! A pure-Rust HDF5 library, built for speed
//!
//! This library does not intend to support all features of HDF5 either in the library or the
//! specification.

use std::collections::BTreeMap; // Currently use BTreeMap just to get sorted Debug output
use std::path::Path;

mod error;
mod parse;
pub use error::Error;

/// Convienence function for Hdf5File::read
pub fn read<P: AsRef<Path>>(path: P) -> Result<Hdf5File, Error> {
    Hdf5File::read(path)
}

/// An opened HDF5 file
#[derive(Debug)]
pub struct Hdf5File {
    map: memmap::Mmap,
    root_group: Group,
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

#[derive(Debug)]
struct Attribute {
    dtype: Hdf5Dtype,
    #[allow(dead_code)]
    dimensions: Vec<u64>,
    data: Vec<u8>,
}

impl Attribute {
    fn from(parsed: parse::header::Attribute) -> Self {
        Attribute {
            dtype: Hdf5Dtype::from(parsed.datatype),
            dimensions: parsed.dataspace.dimensions,
            data: parsed.data,
        }
    }
}

impl std::fmt::Display for Attribute {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.dtype {
            Hdf5Dtype::I32 => write!(f, "{}", i32::from_bytes(&self.data)),
            Hdf5Dtype::I64 => write!(f, "{}", i64::from_bytes(&self.data)),
            Hdf5Dtype::F32 => write!(f, "{}", f32::from_bytes(&self.data)),
            Hdf5Dtype::F64 => write!(f, "{}", f64::from_bytes(&self.data)),
            Hdf5Dtype::String(s) => {
                write!(f, "{:?}", String::from_utf8_lossy(&self.data[..s as usize]))
            }
            Hdf5Dtype::Bool => write!(f, "{}", self.data[0] > 0),
        }
    }
}

struct Dataset {
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

/// Identifies Rust types that this library can produce from HDF5 types
pub trait Hdf5Type: private::Sealed {
    fn dtype() -> Hdf5Dtype;
    fn from_bytes(bytes: &[u8]) -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

impl Hdf5Type for i32 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::I32
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl Hdf5Type for i64 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::I64
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

impl Hdf5Type for f32 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::F32
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_bits(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

impl Hdf5Type for f64 {
    fn dtype() -> Hdf5Dtype {
        Hdf5Dtype::F64
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_bits(u64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hdf5Dtype {
    F64,
    F32,
    I64,
    I32,
    String(u32),
    Bool,
}

impl Hdf5Dtype {
    fn from(raw: parse::header::DataType) -> Self {
        use parse::header::DatatypeClass;
        match (raw.class.clone(), raw.size.clone()) {
            (DatatypeClass::FixedPoint, 4) => Hdf5Dtype::I32,
            (DatatypeClass::FixedPoint, 8) => Hdf5Dtype::I64,
            (DatatypeClass::FloatingPoint, 8) => Hdf5Dtype::F64,
            (DatatypeClass::FloatingPoint, 4) => Hdf5Dtype::F32,
            (DatatypeClass::String, s) => Hdf5Dtype::String(s),
            (DatatypeClass::Enumerated, 1) => Hdf5Dtype::Bool,
            _ => unimplemented!("dtype not supported yet {:#?}", raw),
        }
    }
}

impl Hdf5File {
    /// Open an HDF5 file
    ///
    /// This function memory-maps the file and initializes a number of internal data structures to
    /// make access to data trivial.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = std::fs::File::open(path)?;
        let contents = unsafe { memmap::Mmap::map(&file)? };
        let superblock = parse::superblock(&contents)?.1;

        let (mut remaining, object_header) = parse::object_header(
            &contents[superblock
                .root_group_symbol_table_entry
                .object_header_address as usize..],
        )?;

        use std::ops::Deref;
        let mut root_group = parse_group(
            contents.deref(),
            parse::header::SymbolTable {
                btree_address: superblock.root_group_symbol_table_entry.address_of_btree,
                local_heap_address: superblock
                    .root_group_symbol_table_entry
                    .address_of_name_heap,
            },
        )?;

        use parse::header::Message;
        use parse::header::ObjectHeaderContinuation;
        let mut messages = Vec::new();
        for _ in 0..object_header.total_number_of_header_messages {
            let (rem, message) = parse::header_message(remaining)?;
            match message {
                Message::Attribute(m) => {
                    root_group
                        .attributes
                        .insert(m.name.clone(), Attribute::from(m));
                    remaining = rem;
                }
                Message::ObjectHeaderContinuation(ObjectHeaderContinuation { offset, .. }) => {
                    remaining = &contents[offset as usize..];
                }
                _ => {
                    messages.push(message.clone());
                    remaining = rem;
                }
            }
        }

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

    pub fn attr<T: Hdf5Type>(&self, attribute_path: &str) -> T {
        let attribute = self
            .root_group
            .attributes
            .get(attribute_path)
            .expect(&format!("attribute not found: {:?}", attribute_path));
        if T::dtype() != attribute.dtype {
            panic!(
                "Attribute {:?} is of type {:?}, not {:?}",
                attribute_path,
                attribute.dtype,
                T::dtype()
            );
        }
        T::from_bytes(&attribute.data)
    }
}

fn parse_group(contents: &[u8], symbol_table: parse::header::SymbolTable) -> Result<Group, Error> {
    let node = parse::hdf5_node(&contents[symbol_table.btree_address as usize..], 8)?.1;

    let name_heap =
        parse::local_heap(&contents[symbol_table.local_heap_address as usize..], 8, 8)?.1;

    let mut datasets = BTreeMap::new();
    let mut groups = BTreeMap::new();
    let mut attributes = BTreeMap::new();

    for group_entry in node.entries {
        let table =
            parse::symbol_table(&contents[group_entry.pointer_to_symbol_table as usize..], 8)?.1;

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
                parse::object_header(&contents[object.object_header_address as usize..])?;
            let mut messages = Vec::new();

            for _ in 0..object_header.total_number_of_header_messages {
                let (rem, message) = parse::header_message(remaining)?;
                if let Message::ObjectHeaderContinuation(ObjectHeaderContinuation {
                    offset, ..
                }) = message
                {
                    remaining = &contents[offset as usize..];
                } else {
                    messages.push(message.clone());
                    remaining = rem;
                }
            }

            // Datasets have a data layout, data storage, datatype, and dataspace
            let mut layout = None;
            let mut fillvalue = None;
            let mut dtype = None;
            let mut dspace = None;
            let mut symbol_table = None;

            for message in messages {
                match message {
                    Message::DataLayout(m) => layout = Some(m),
                    Message::DataStorageFillValue(m) => fillvalue = Some(m),
                    Message::DataType(m) => dtype = Some(m),
                    Message::Dataspace(m) => dspace = Some(m),
                    Message::Attribute(m) => {
                        attributes.insert(m.name.clone(), Attribute::from(m));
                    }
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
                _ => panic!("Found an HDF5 object that is not a dataset or a group"),
            }
        }
    }

    Ok(Group {
        datasets,
        attributes,
        groups,
    })
}
