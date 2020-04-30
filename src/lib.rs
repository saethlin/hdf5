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
        let delim_index = dataset_path.find('/');
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

    //fn find_group(&self, dataset_path: &str) -> &Dataset {}
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
        Self {
            dtype: Hdf5Dtype::from(parsed.datatype),
            dimensions: parsed.dataspace.dimensions,
            data: parsed.data,
        }
    }
}

#[derive(Debug)]
struct Dataset {
    dimensions: Vec<u64>,
    dtype: Hdf5Dtype,
    address: u64,
    size: u64,
    attributes: BTreeMap<String, Attribute>,
}

impl Dataset {
    fn from(messages: Vec<parse::header::Message>) -> Self {
        use parse::header::Message;
        let mut dimensions = None;
        let mut dtype = None;
        let mut address = None;
        let mut size = None;
        let mut attributes = BTreeMap::new();
        for message in messages {
            match message {
                Message::DataLayout(m) => {
                    address = Some(m.address);
                    size = Some(m.size);
                }
                Message::DataType(m) => dtype = Some(Hdf5Dtype::from(m)),
                Message::Dataspace(m) => dimensions = Some(m.dimensions),
                Message::Attribute(m) => {
                    attributes.insert(m.name.clone(), Attribute::from(m));
                }
                Message::DataStorageFillValue(_) => {}
                Message::ObjectModificationTime(_) => {}
                Message::Nil => {}
                m => unimplemented!("Unexpected message for a Dataset {:?}", m),
            }
        }

        let dimensions = dimensions.unwrap();
        let dtype = dtype.unwrap();
        let address = address.unwrap();
        let size = size.unwrap();

        Self {
            dimensions,
            dtype,
            address,
            size,
            attributes,
        }
    }
}

/// Identifies Rust types that this library can produce from HDF5 types
pub trait FromHdf5: private::Sealed {
    fn from_types() -> &'static [Hdf5Dtype];
    // We need the first two fields for VlenString
    // so that we can convert the global heap ID
    fn convert(file: &Hdf5File, dtype: Hdf5Dtype, data: &[u8]) -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for String {}
}

impl FromHdf5 for i32 {
    fn from_types() -> &'static [Hdf5Dtype] {
        &[Hdf5Dtype::I32]
    }

    fn convert(_: &Hdf5File, _: Hdf5Dtype, bytes: &[u8]) -> Self {
        Self::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl FromHdf5 for i64 {
    fn from_types() -> &'static [Hdf5Dtype] {
        &[Hdf5Dtype::I64]
    }

    fn convert(_: &Hdf5File, _: Hdf5Dtype, data: &[u8]) -> Self {
        Self::from_ne_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ])
    }
}

impl FromHdf5 for f32 {
    fn from_types() -> &'static [Hdf5Dtype] {
        &[Hdf5Dtype::F32]
    }

    fn convert(_: &Hdf5File, _: Hdf5Dtype, data: &[u8]) -> Self {
        Self::from_ne_bytes([data[0], data[1], data[2], data[3]])
    }
}

impl FromHdf5 for f64 {
    fn from_types() -> &'static [Hdf5Dtype] {
        &[Hdf5Dtype::F64]
    }

    fn convert(_: &Hdf5File, _: Hdf5Dtype, data: &[u8]) -> Self {
        Self::from_ne_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ])
    }
}

impl FromHdf5 for String {
    fn from_types() -> &'static [Hdf5Dtype] {
        &[Hdf5Dtype::String, Hdf5Dtype::VlenString]
    }

    fn convert(file: &Hdf5File, dtype: Hdf5Dtype, data: &[u8]) -> Self {
        match dtype {
            Hdf5Dtype::String => Self::from_utf8_lossy(data).into_owned(),
            Hdf5Dtype::VlenString => {
                assert_eq!(data.len(), 16);
                let data = &data[4..];
                let heap_address = u64::from_ne_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]) as usize;
                let heap_index = u16::from_ne_bytes([data[8], data[9]]);
                let heap_object =
                    parse::global_heap_nth_item(&file.map[heap_address..], heap_index)
                        .unwrap()
                        .1;
                Self::from_utf8_lossy(heap_object).into_owned()
            }
            _ => unreachable!(),
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hdf5Dtype {
    F64,
    F32,
    I64,
    I32,
    String,
    VlenString,
    Bool,
}

impl Hdf5Dtype {
    fn from(raw: parse::header::DataType) -> Self {
        use parse::header::DatatypeClass;
        match (raw.class.clone(), raw.size) {
            (DatatypeClass::FixedPoint, 4) => Self::I32,
            (DatatypeClass::FixedPoint, 8) => Self::I64,
            (DatatypeClass::FloatingPoint, 8) => Self::F64,
            (DatatypeClass::FloatingPoint, 4) => Self::F32,
            (DatatypeClass::String, _) => Self::String,
            (DatatypeClass::Enumerated, 1) => Self::Bool,
            (
                DatatypeClass::VariableLength {
                    ty: 1, padding: 0, ..
                },
                _,
            ) => Self::VlenString,
            _ => unimplemented!("dtype not supported yet {:?}", raw),
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
        let mut resume_with_after_continuation = Vec::new();
        for _ in 0..object_header.total_number_of_header_messages {
            let (remaining_after_parse, message) = parse::header_message(remaining)?;
            match message {
                Message::Attribute(m) => {
                    root_group
                        .attributes
                        .insert(m.name.clone(), Attribute::from(m));
                    remaining = remaining_after_parse;
                }
                Message::ObjectHeaderContinuation(ObjectHeaderContinuation { offset, length }) => {
                    resume_with_after_continuation.push(remaining_after_parse);
                    remaining = &contents[offset as usize..offset as usize + length as usize];
                }
                _ => {
                    messages.push(message.clone());
                    remaining = remaining_after_parse;
                    if remaining.is_empty() {
                        remaining = resume_with_after_continuation
                            .pop()
                            .expect("Ran out of data to parse, and no contiuation to resume from");
                    }
                }
            }
        }

        Ok(Self {
            map: contents,
            root_group,
        })
    }

    /// Look up the provided path to a dataset, if one is found and its type correct, return a
    /// slice of the underlying file mapping.
    ///
    /// Note that this discards any dimension information associated with the dataset.
    pub fn view(&self, dataset_path: &str) -> &[u8] {
        let dataset = self.root_group.find_dataset(dataset_path);
        &self.map[dataset.address as usize..(dataset.address + dataset.size) as usize]
    }

    /// Look up the provided path to an attribute, if one is found and its type correct,
    /// return a copy of the attribute's data.
    ///
    /// Panics if the attribute cannot be found or the attribute is of the wrong type.
    pub fn attr<T: FromHdf5>(&self, attribute_name: &str) -> T {
        let attribute = self
            .root_group
            .attributes
            .get(attribute_name)
            .unwrap_or_else(|| panic!("attribute not found: {:?}", attribute_name));
        if !T::from_types().contains(&attribute.dtype) {
            panic!(
                "Attribute {:?} is of type {:?}, which is not compatible with any of {:?}",
                attribute_name,
                attribute.dtype,
                T::from_types()
            );
        }
        T::convert(self, attribute.dtype, &attribute.data)
    }
}

fn parse_group(contents: &[u8], symbol_table: parse::header::SymbolTable) -> Result<Group, Error> {
    let node = parse::hdf5_node(&contents[symbol_table.btree_address as usize..], 8)?.1;

    let name_heap =
        parse::local_heap(&contents[symbol_table.local_heap_address as usize..], 8, 8)?.1;

    let mut datasets = BTreeMap::new();
    let mut groups = BTreeMap::new();

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

            let mut resume_with_after_continuation = Vec::new();
            for _ in 0..object_header.total_number_of_header_messages {
                let (remaining_after_parse, message) = parse::header_message(remaining)?;
                if let Message::ObjectHeaderContinuation(ObjectHeaderContinuation {
                    offset,
                    length,
                }) = message
                {
                    resume_with_after_continuation.push(remaining_after_parse);
                    remaining = &contents[offset as usize..offset as usize + length as usize];
                } else {
                    messages.push(message.clone());
                    remaining = remaining_after_parse;
                    if remaining.is_empty() {
                        remaining = resume_with_after_continuation
                            .pop()
                            .expect("Ran out of data to parse, and no contiuation to resume from");
                    }
                }
            }

            match messages.first() {
                None => {}
                Some(Message::SymbolTable(table)) => {
                    groups.insert(name.clone(), parse_group(contents, table.clone())?);
                    let g = groups.get_mut(name).unwrap();
                    for m in messages.into_iter().skip(1) {
                        if let Message::Attribute(m) = m {
                            g.attributes.insert(m.name.clone(), Attribute::from(m));
                        }
                    }
                }
                Some(Message::Dataspace(_)) => {
                    datasets.insert(name.clone(), Dataset::from(messages));
                }
                m => unimplemented!("Unexpected first message in a message list: {:?}", m),
            };
        }
    }

    Ok(Group {
        datasets,
        groups,
        attributes: BTreeMap::new(),
    })
}
