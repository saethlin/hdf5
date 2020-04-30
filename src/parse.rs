use nom::bytes::streaming::{tag, take};
use nom::error::context;
use nom::multi::count;
use nom::number::streaming::{le_u16, le_u24, le_u32, le_u64, le_u8};

type Result<'a, O> =
    std::result::Result<(&'a [u8], O), nom::Err<nom::error::VerboseError<&'a [u8]>>>;

#[derive(Debug)]
pub struct Hdf5Superblock {
    pub superblock_version: u8,
    pub free_space_storage_version: u8,
    pub root_group_symbol_table_entry_version: u8,
    pub shared_header_message_format_version: u8,
    pub offset_size: u8,
    pub length_size: u8,
    pub group_leaf_node_k: u16,
    pub group_internal_node_k: u16,
    pub file_consistency_flags: u32,
    pub base_address: u64,
    pub address_of_file_free_space_info: u64,
    pub end_of_file_address: u64,
    pub driver_information_block_address: u64,
    pub root_group_symbol_table_entry: SymbolTableEntry,
}

fn address<'a>(len: u8) -> impl Fn(&'a [u8]) -> Result<u64> {
    nom::combinator::map_parser(take(len), le_u64)
}

pub fn superblock(input: &[u8]) -> Result<Hdf5Superblock> {
    context("superblock", |input| {
        let (input, _) = tag(b"\x89\x48\x44\x46\x0d\x0a\x1a\x0a")(input)?;
        let (input, superblock_version) = le_u8(input)?;
        let (input, free_space_storage_version) = le_u8(input)?;
        let (input, root_group_symbol_table_entry_version) = le_u8(input)?;
        let (input, _) = tag([0])(input)?;
        let (input, shared_header_message_format_version) = le_u8(input)?;
        let (input, offset_size) = le_u8(input)?;
        let (input, length_size) = le_u8(input)?;
        let (input, _) = tag([0])(input)?;
        let (input, group_leaf_node_k) = le_u16(input)?;
        let (input, group_internal_node_k) = le_u16(input)?;
        let (input, file_consistency_flags) = le_u32(input)?;
        let (input, base_address) = address(offset_size)(input)?;
        let (input, address_of_file_free_space_info) = address(offset_size)(input)?;
        let (input, end_of_file_address) = address(offset_size)(input)?;
        let (input, driver_information_block_address) = address(offset_size)(input)?;
        let (input, root_group_symbol_table_entry) = symbol_table_entry(input, offset_size)?;

        Ok((
            input,
            Hdf5Superblock {
                superblock_version,
                free_space_storage_version,
                root_group_symbol_table_entry_version,
                shared_header_message_format_version,
                offset_size,
                length_size,
                group_leaf_node_k,
                group_internal_node_k,
                file_consistency_flags,
                base_address,
                address_of_file_free_space_info,
                end_of_file_address,
                driver_information_block_address,
                root_group_symbol_table_entry,
            },
        ))
    })(input)
}

#[derive(Debug)]
pub struct SymbolTable {
    pub version: u8,
    pub entries: Vec<SymbolTableEntry>,
}

pub fn symbol_table(input: &[u8], offset_size: u8) -> Result<SymbolTable> {
    context("symbol table", |input| {
        let (input, _) = tag(b"SNOD")(input)?;
        let (input, version) = le_u8(input)?;
        let (input, _) = tag([0])(input)?;
        let (input, number_of_symbols) = le_u16(input)?;
        let (input, entries) = count(
            |i| symbol_table_entry(i, offset_size),
            number_of_symbols as usize,
        )(input)?;
        Ok((input, SymbolTable { version, entries }))
    })(input)
}

#[derive(Debug)]
pub struct SymbolTableEntry {
    pub link_name_offset: u64,
    pub object_header_address: u64,
    pub cache_type: u32,
    pub address_of_btree: u64,
    pub address_of_name_heap: u64,
}

pub fn symbol_table_entry(input: &[u8], offset_size: u8) -> Result<SymbolTableEntry> {
    context("symbol table entry", |input| {
        let (
            input,
            (
                link_name_offset,
                object_header_address,
                cache_type,
                _,
                address_of_btree,
                address_of_name_heap,
            ),
        ) = nom::sequence::tuple((
            address(offset_size),
            address(offset_size),
            le_u32,
            tag([0, 0, 0, 0]),
            address(offset_size),
            address(offset_size),
        ))(input)?;

        Ok((
            input,
            SymbolTableEntry {
                link_name_offset,
                object_header_address,
                cache_type,
                address_of_btree,
                address_of_name_heap,
            },
        ))
    })(input)
}

#[derive(Debug, Clone)]
pub struct GroupEntry {
    pub byte_offset_into_local_heap: u64,
    pub pointer_to_symbol_table: u64,
}

pub fn group_entry(input: &[u8]) -> Result<GroupEntry> {
    context("group entry", |input| {
        let (input, byte_offset_into_local_heap) = le_u64(input)?;
        let (input, pointer_to_symbol_table) = le_u64(input)?;
        Ok((
            input,
            GroupEntry {
                byte_offset_into_local_heap,
                pointer_to_symbol_table,
            },
        ))
    })(input)
}

#[derive(Debug)]
pub struct GroupNode {
    pub node_level: u8,
    pub entries_used: u16,
    pub address_of_left_sibling: u64,
    pub address_of_right_sibling: u64,
    pub entries: Vec<GroupEntry>,
}

pub fn hdf5_node(input: &[u8], offset_size: u8) -> Result<GroupNode> {
    context("HDF5 node", |input| {
        let (input, _) = tag(b"TREE")(input)?;
        let (input, _) = tag([0])(input)?; // We only support group nodes
        group_node(input, offset_size)
    })(input)
}

pub fn group_node(input: &[u8], offset_size: u8) -> Result<GroupNode> {
    context("group node", |input| {
        let (input, node_level) = le_u8(input)?;
        let (input, entries_used) = le_u16(input)?;
        let (input, address_of_left_sibling) = address(offset_size)(input)?;
        let (input, address_of_right_sibling) = address(offset_size)(input)?;
        let (input, entries) = count(group_entry, entries_used as usize)(input)?;

        Ok((
            input,
            GroupNode {
                node_level,
                entries_used,
                address_of_left_sibling,
                address_of_right_sibling,
                entries,
            },
        ))
    })(input)
}

#[derive(Debug)]
pub struct LocalHeap {
    pub version: u8,
    pub data_segment_size: u64,
    pub offset_to_head_of_freelist: u64,
    pub address_of_data_segment: u64,
}

pub fn local_heap(input: &[u8], offset_size: u8, length_size: u8) -> Result<LocalHeap> {
    context("local heap", |input| {
        let (input, _) = tag(b"HEAP")(input)?;
        let (input, version) = le_u8(input)?;
        let (input, _) = tag([0, 0, 0])(input)?;
        let (input, data_segment_size) = address(length_size)(input)?;
        let (input, offset_to_head_of_freelist) = address(length_size)(input)?;
        let (input, address_of_data_segment) = address(offset_size)(input)?;

        Ok((
            input,
            LocalHeap {
                version,
                data_segment_size,
                offset_to_head_of_freelist,
                address_of_data_segment,
            },
        ))
    })(input)
}

#[derive(Debug)]
pub struct ObjectHeader {
    pub version: u8,
    pub total_number_of_header_messages: u16,
    pub object_reference_count: u32,
    pub object_header_size: u32,
}

pub fn object_header(input: &[u8]) -> Result<ObjectHeader> {
    context("object header", |input| {
        let (input, version) = le_u8(input)?;
        let (input, _) = tag([0])(input)?;
        let (input, total_number_of_header_messages) = le_u16(input)?;
        let (input, object_reference_count) = le_u32(input)?;
        let (input, object_header_size) = le_u32(input)?;
        // Pad to 8-byte alignment?
        let (input, _) = take(4usize)(input)?;
        Ok((
            input,
            ObjectHeader {
                version,
                total_number_of_header_messages,
                object_reference_count,
                object_header_size,
            },
        ))
    })(input)
}

pub mod header {
    #[derive(Debug, Clone)]
    pub struct Dataspace {
        pub dimensionality: u8,
        pub flags: u8,
        pub dimensions: Vec<u64>,
        pub max_dimensions: Option<Vec<u64>>,
    }

    #[derive(Debug, Clone)]
    pub enum DatatypeClass {
        FixedPoint,
        FloatingPoint,
        Time,
        String,
        Bitfield,
        Opaque,
        Compound,
        Reference,
        Enumerated,
        VariableLength {
            ty: u8,
            padding: u8,
            character_set: u8,
        },
        Array,
    }

    #[derive(Debug, Clone)]
    pub struct DataType {
        pub version: u8,
        pub class: DatatypeClass,
        pub class_bitfields: u32,
        pub size: u32,
        pub properties: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    pub struct DataStorageFillValue {
        pub space_allocation_time: u8,
        pub fill_value_write_time: u8,
        pub fill_value_defined: u8,
        pub size: u32,
        pub fill_value: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    pub struct DataLayout {
        pub address: u64,
        pub size: u64,
    }

    #[derive(Debug, Clone)]
    pub struct Attribute {
        pub datatype: DataType,
        pub dataspace: Dataspace,
        pub data: Vec<u8>,
        pub name: String,
    }

    #[derive(Debug, Clone)]
    pub struct ObjectHeaderContinuation {
        pub offset: u64,
        pub length: u64,
    }

    #[derive(Debug, Clone)]
    pub struct SymbolTable {
        pub btree_address: u64,
        pub local_heap_address: u64,
    }

    #[derive(Debug, Clone)]
    pub struct ObjectModificationTime {
        pub seconds_after_unix_epoch: u32,
    }

    #[derive(Debug, Clone)]
    pub enum Message {
        Nil,
        Dataspace(Dataspace),
        //LinkInfo,
        DataType(DataType),
        DataStorageFillValue(DataStorageFillValue),
        /*
        Link,
        DataStorageExternal,
        */
        DataLayout(DataLayout),
        Attribute(Attribute),
        /*
        ObjectComment,
        SharedMessageTable,
        */
        ObjectHeaderContinuation(ObjectHeaderContinuation),
        SymbolTable(SymbolTable),
        ObjectModificationTime(ObjectModificationTime),
        /*
        BtreeKValues,
        DriverInfo,
        AttributeInfo,
        ObjectReferenceCount,
        */
    }
}

fn datatype(input: &[u8], message_size: u16) -> Result<header::DataType> {
    use header::DatatypeClass::*;
    context("datatype", |input| {
        let (input, class_and_version) = le_u8(input)?;
        let (input, class_bitfields) = le_u24(input)?;
        let (input, size) = le_u32(input)?;
        let (input, properties) = count(le_u8, message_size as usize - 8)(input)?;

        let version = class_and_version >> 4;
        let raw_class = class_and_version & 0b0000_1111;
        let class = match raw_class {
            0 => FixedPoint,
            1 => FloatingPoint,
            2 => Time,
            3 => header::DatatypeClass::String,
            4 => Bitfield,
            5 => Opaque,
            6 => Compound,
            7 => Reference,
            8 => Enumerated,
            9 => VariableLength {
                ty: (class_bitfields & 0b111) as u8,
                padding: (class_bitfields >> 3 & 0b111) as u8,
                character_set: (class_bitfields >> 8 & 0b111) as u8,
            },
            10 => Array,
            _ => panic!("Invalid datatype class: {}", raw_class),
        };

        Ok((
            input,
            header::DataType {
                version,
                class,
                class_bitfields,
                size,
                properties,
            },
        ))
    })(input)
}

fn dataspace(input: &[u8]) -> Result<header::Dataspace> {
    context("dataspace", |input| {
        let (input, _) = tag([1])(input)?;
        let (input, dimensionality) = le_u8(input)?;
        let (input, flags) = le_u8(input)?;
        let (input, _ty) = le_u8(input)?;
        // Eat the unused bytes in version 1
        let (input, _) = nom::bytes::streaming::take(4usize)(input)?;
        let (input, (dimensions, max_dimensions)) = if flags == 0 {
            let (input, dimensions) = count(le_u64, dimensionality as usize)(input)?;
            (input, (dimensions, None))
        } else if flags == 1 {
            let (input, dimensions) = count(le_u64, dimensionality as usize)(input)?;
            let (input, max_dimensions) = count(le_u64, dimensionality as usize)(input)?;
            (input, (dimensions, Some(max_dimensions)))
        } else {
            unimplemented!("Permutation indices are not supported");
        };

        Ok((
            input,
            header::Dataspace {
                dimensionality,
                flags,
                dimensions,
                max_dimensions,
            },
        ))
    })(input)
}

pub fn fill_value(input: &[u8]) -> Result<header::DataStorageFillValue> {
    context("fill value", |input| {
        let (input, version) = le_u8(input)?; // Only support version 1
        if version == 2 {
            let (input, space_allocation_time) = le_u8(input)?;
            let (input, fill_value_write_time) = le_u8(input)?;
            let (input, fill_value_defined) = le_u8(input)?;
            let (input, size) = if fill_value_defined > 0 {
                le_u32(input)?
            } else {
                (input, 0)
            };
            let (input, fill_value) = if fill_value_defined > 0 {
                count(le_u8, size as usize)(input)?
            } else {
                (input, Vec::new())
            };
            Ok((
                input,
                header::DataStorageFillValue {
                    space_allocation_time,
                    fill_value_write_time,
                    fill_value_defined,
                    size,
                    fill_value,
                },
            ))
        } else {
            unimplemented!("Unsupported DataStorageFillValue version {}", version);
        }
    })(input)
}

pub fn data_layout(input: &[u8]) -> Result<header::DataLayout> {
    context("data layout", |input| {
        let (input, version) = le_u8(input)?;
        if version != 3 {
            unimplemented!("Unsupported DataLayout version {}", version);
        }
        let (input, layout_class) = le_u8(input)?;
        if layout_class != 1 {
            unimplemented!("Unsupported DataLayout class {}", layout_class);
        }
        let (input, data_address) = address(8)(input)?;
        let (input, size) = address(8)(input)?;
        let (input, _) = take(6usize)(input)?;

        Ok((
            input,
            header::DataLayout {
                address: data_address,
                size,
            },
        ))
    })(input)
}

fn attribute(input: &[u8], message_size: u16) -> Result<header::Attribute> {
    context("attribute", |input| {
        let (input, _) = tag([1])(input)?;
        let (input, _) = tag([0])(input)?;
        let (input, name_size) = le_u16(input)?;
        let (input, datatype_size) = le_u16(input)?;
        let (input, dataspace_size) = le_u16(input)?;

        let (_, name) = nom::bytes::streaming::take(name_size)(input)?;
        let name =
            String::from_utf8(name.iter().take_while(|b| **b > 0).copied().collect()).unwrap();
        let input = &input[pad8(name_size)..];

        let (_, datatype) = datatype(input, datatype_size)?;
        let input = &input[pad8(datatype_size)..];

        let (_, dataspace) = dataspace(input)?;
        let input = &input[pad8(dataspace_size)..];

        let data_len = message_size as usize
            - (8 + pad8(name_size) + pad8(datatype_size) + pad8(dataspace_size));
        let (input, data) = nom::bytes::streaming::take(data_len)(input)?;

        Ok((
            input,
            header::Attribute {
                name,
                datatype,
                dataspace,
                data: data.to_vec(),
            },
        ))
    })(input)
}

pub fn object_header_continuation(input: &[u8]) -> Result<header::ObjectHeaderContinuation> {
    context("object header continuation", |input| {
        let (input, (offset, length)) = nom::sequence::tuple((address(8), address(8)))(input)?;
        Ok((input, header::ObjectHeaderContinuation { length, offset }))
    })(input)
}

pub fn symbol_table_message(input: &[u8]) -> Result<header::SymbolTable> {
    context("symbol table message", |input| {
        let (input, btree_address) = address(8)(input)?;
        let (input, local_heap_address) = address(8)(input)?;

        Ok((
            input,
            header::SymbolTable {
                btree_address,
                local_heap_address,
            },
        ))
    })(input)
}

pub fn object_modification_time(input: &[u8]) -> Result<header::ObjectModificationTime> {
    context("object modification time", |input| {
        let (input, _) = tag([1])(input)?; // version 1 is the only allowed by the standard
        let (input, _) = tag([0, 0, 0])(input)?; // padding
        let (input, seconds) = le_u32(input)?;
        Ok((
            input,
            header::ObjectModificationTime {
                seconds_after_unix_epoch: seconds,
            },
        ))
    })(input)
}

pub fn header_message(input: &[u8]) -> Result<header::Message> {
    context("header message", |input| {
        let (input, message_type) = le_u16(input)?;
        let (input, message_size) = le_u16(input)?;
        let (input, _flags) = le_u8(input)?;
        let (input, _) = tag([0, 0, 0])(input)?;
        use header::Message;
        use nom::combinator::map;
        match message_type {
            0x0 => Ok((input, header::Message::Nil)),
            0x1 => map(dataspace, Message::Dataspace)(input),
            0x3 => datatype(input, message_size).map(|(i, dtype)| (i, Message::DataType(dtype))),
            0x5 => map(fill_value, Message::DataStorageFillValue)(input),
            0x8 => map(data_layout, Message::DataLayout)(input),
            0xC => attribute(input, message_size).map(|(i, attr)| (i, Message::Attribute(attr))),
            0x10 => map(
                object_header_continuation,
                Message::ObjectHeaderContinuation,
            )(input),
            0x11 => map(symbol_table_message, Message::SymbolTable)(input),
            0x12 => map(object_modification_time, Message::ObjectModificationTime)(input),
            _ => {
                unimplemented!("unknown header message {:04X}", message_type);
            }
        }
    })(input)
}

pub fn global_heap_nth_item(input: &[u8], desired_index: u16) -> Result<&[u8]> {
    context("global heap", |input| {
        let (input, _) = tag(b"GCOL")(input)?;
        let (input, _) = tag([1])(input)?; // Only version 1 exists
        let (input, _) = tag([0, 0, 0])(input)?; // Reserved zero bytes
        let (input, _collection_size) = address(8)(input)?;

        loop {
            // Parse the heap object and check if it's what we are looking for
            let (input, heap_object_index) = le_u16(input)?;
            let (input, _reference_count) = le_u16(input)?;
            let (input, _) = tag([0, 0, 0, 0])(input)?;
            let (input, object_size) = address(8)(input)?;
            let (input, object_data) = take(object_size)(input)?;
            if heap_object_index == desired_index {
                break Ok((input, object_data));
            }
        }
    })(input)
}

fn pad8<T>(t: T) -> usize
where
    usize: From<T>,
{
    let t = usize::from(t);
    if t % 8 == 0 {
        t
    } else {
        t + (8 - (t % 8))
    }
}
