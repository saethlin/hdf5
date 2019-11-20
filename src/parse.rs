use nom::{call, cond, count, dbg, do_parse, named, named_args, tag, take};

use nom::bytes::streaming::{tag, take};
use nom::multi::count;
use nom::number::streaming::{le_u16, le_u24, le_u32, le_u64, le_u8};
use nom::IResult;

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

fn address<'a>(len: u8) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], u64> {
    nom::combinator::map_parser(take(len), le_u64)
}

pub fn superblock(input: &[u8]) -> IResult<&[u8], Hdf5Superblock> {
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
}

#[derive(Debug)]
pub struct SymbolTable {
    pub version: u8,
    pub entries: Vec<SymbolTableEntry>,
}

pub fn symbol_table(input: &[u8], offset_size: u8) -> IResult<&[u8], SymbolTable> {
    let (input, _) = tag(b"SNOD")(input)?;
    let (input, version) = le_u8(input)?;
    let (input, _) = tag([0])(input)?;
    let (input, number_of_symbols) = le_u16(input)?;
    let (input, entries) = count(
        |i| symbol_table_entry(i, offset_size),
        number_of_symbols as usize,
    )(input)?;
    Ok((input, SymbolTable { version, entries }))
}

#[derive(Debug)]
pub struct SymbolTableEntry {
    pub link_name_offset: u64,
    pub object_header_address: u64,
    pub cache_type: u32,
    pub address_of_btree: u64,
    pub address_of_name_heap: u64,
}

pub fn symbol_table_entry(input: &[u8], offset_size: u8) -> IResult<&[u8], SymbolTableEntry> {
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
}

#[derive(Debug, Clone)]
pub struct GroupEntry {
    pub byte_offset_into_local_heap: u64,
    pub pointer_to_symbol_table: u64,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(pub group_entry<GroupEntry>,
    do_parse!(
        byte_offset_into_local_heap: le_u64 >>
        pointer_to_symbol_table: le_u64 >>
        (GroupEntry {
            byte_offset_into_local_heap,
            pointer_to_symbol_table,
        })
    )
);

#[derive(Debug)]
pub struct GroupNode {
    pub node_level: u8,
    pub entries_used: u16,
    pub address_of_left_sibling: u64,
    pub address_of_right_sibling: u64,
    pub entries: Vec<GroupEntry>,
}

named_args!(pub hdf5_node (offset_size: u8) <GroupNode>,
    do_parse!(
        tag!(b"TREE") >>
        tag!([0]) >> // We only support group nodes
        node: call!(group_node, offset_size) >>
        (node)
    )
);

pub fn group_node(input: &[u8], offset_size: u8) -> IResult<&[u8], GroupNode> {
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
}

#[derive(Debug)]
pub struct LocalHeap {
    pub version: u8,
    pub data_segment_size: u64,
    pub offset_to_head_of_freelist: u64,
    pub address_of_data_segment: u64,
}

pub fn local_heap(input: &[u8], offset_size: u8, length_size: u8) -> IResult<&[u8], LocalHeap> {
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
}

#[derive(Debug)]
pub struct ObjectHeader {
    pub version: u8,
    pub total_number_of_header_messages: u16,
    pub object_reference_count: u32,
    pub object_header_size: u32,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(pub object_header <ObjectHeader>,
    do_parse!(
        version: le_u8 >>
        dbg!(tag!(b"\0")) >>
        total_number_of_header_messages: le_u16 >>
        object_reference_count: le_u32 >>
        object_header_size: le_u32 >>
        take!(4) >> // This needs to be variable, to ensure that we are 8-byte aligned
        (ObjectHeader {
            version,
            total_number_of_header_messages,
            object_reference_count,
            object_header_size,
        })
    )
);

pub mod header {
    #[derive(Debug, Clone)]
    pub struct Dataspace {
        pub version: u8,
        pub dimensionality: u8,
        pub flags: u8,
        pub dimensions: Vec<u64>,
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
        pub version: u8,
        pub space_allocation_time: u8,
        pub fill_value_write_time: u8,
        pub fill_value_defined: u8,
        pub size: Option<u32>,
        pub fill_value: Option<Vec<u8>>,
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

fn datatype(input: &[u8], message_size: u16) -> IResult<&[u8], header::DataType> {
    use header::DatatypeClass::*;
    let (input, class_and_version) = le_u8(input)?;
    let (input, class_bitfields) = le_u24(input)?;
    let (input, size) = le_u32(input)?;
    let (input, properties) = count(le_u8, message_size as usize - 8)(input)?;

    let version = class_and_version >> 4;
    let class = match class_and_version & 0b0000_1111 {
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
        _ => panic!("unknown dtype"),
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
}

fn dataspace(input: &[u8]) -> IResult<&[u8], header::Dataspace> {
    let (input, (version, dimensionality, flags)) =
        nom::sequence::tuple((le_u8, le_u8, le_u8))(input)?;
    let (input, _) = nom::bytes::streaming::take(5usize)(input)?;
    // address
    let (input, dimensions) = count(le_u64, dimensionality as usize)(input)?;

    Ok((
        input,
        header::Dataspace {
            version,
            dimensionality,
            flags,
            dimensions,
        },
    ))
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(fill_value <header::DataStorageFillValue>,
    do_parse!(
        version: le_u8 >>
        space_allocation_time: le_u8 >>
        fill_value_write_time: le_u8 >>
        fill_value_defined: le_u8 >>
        size: cond!(!(version > 1 && fill_value_defined == 0), le_u32) >>
        fill_value: cond!(size.is_some(), count!(le_u8, size.unwrap() as usize)) >>
        (header::DataStorageFillValue {
            version,
            space_allocation_time,
            fill_value_write_time,
            fill_value_defined,
            size,
            fill_value,
        })
    )
);

pub fn data_layout(input: &[u8]) -> IResult<&[u8], header::DataLayout> {
    let (input, _) = tag([3, 1])(input)?;
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
}

fn attribute(input: &[u8], message_size: u16) -> IResult<&[u8], header::Attribute> {
    let (input, _) = tag([1])(input)?;
    let (input, _) = tag([0])(input)?;
    let (input, name_size) = le_u16(input)?;
    let (input, datatype_size) = le_u16(input)?;
    let (input, dataspace_size) = le_u16(input)?;
    let (input, name) = nom::bytes::streaming::take(pad8(name_size))(input)?;
    let (input, datatype) = datatype(input, datatype_size)?;
    let (input, dataspace) = dataspace(input)?;
    let (input, data) = nom::bytes::streaming::take(
        message_size as usize - (8 + pad8(name_size) + pad8(datatype_size) + pad8(dataspace_size)),
    )(input)?;
    let name = name
        .iter()
        .take_while(|b| **b > 0)
        .map(|b| *b as char)
        .collect();
    // If this is a string, jump to the global heap and read the contents there
    if let header::DataType {
        version: 1,
        class: header::DatatypeClass::VariableLength { .. },
        ..
    } = datatype
    {
        //TODO: Read from the global heap
    }
    Ok((
        input,
        header::Attribute {
            name,
            datatype,
            dataspace,
            data: data.to_vec(),
        },
    ))
}

pub fn object_header_continuation(
    input: &[u8],
) -> IResult<&[u8], header::ObjectHeaderContinuation> {
    let (input, (offset, length)) = nom::sequence::tuple((address(8), address(8)))(input)?;
    Ok((input, header::ObjectHeaderContinuation { length, offset }))
}

pub fn symbol_table_message(input: &[u8]) -> IResult<&[u8], header::SymbolTable> {
    let (input, btree_address) = address(8)(input)?;
    let (input, local_heap_address) = address(8)(input)?;

    Ok((
        input,
        header::SymbolTable {
            btree_address,
            local_heap_address,
        },
    ))
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(object_modification_time <header::ObjectModificationTime>,
    do_parse!(
        dbg!(tag!([1])) >> // version 1 is the only one allowed by the standard
        dbg!(tag!(b"\0\0\0")) >>
        seconds: le_u32 >>
        (header::ObjectModificationTime {
            seconds_after_unix_epoch: seconds,
        })
    )
);

pub fn header_message(input: &[u8]) -> IResult<&[u8], header::Message> {
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
            eprintln!("unknown header message {}", message_type);
            Ok((input, header::Message::Nil))
        }
    }
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
