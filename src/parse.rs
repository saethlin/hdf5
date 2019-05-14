use nom::dbg;
use nom::*;

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

named_args!(address (len: u8) <u64>,
   map!(take!(len), |x| le_u64(x).unwrap().1)
);

named!(pub parse_superblock<&[u8], Hdf5Superblock>,
    do_parse!(
        tag!(b"\x89\x48\x44\x46\x0d\x0a\x1a\x0a") >>
        superblock_version: le_u8 >>
        free_space_storage_version: le_u8 >>
        root_group_symbol_table_entry_version: le_u8 >>
        tag!(b"\0") >>
        shared_header_message_format_version: le_u8 >>
        offset_size: le_u8 >>
        length_size: le_u8 >>
        tag!(b"\0") >>
        group_leaf_node_k: le_u16 >>
        group_internal_node_k: le_u16 >>
        file_consistency_flags: le_u32 >>
        base_address: call!(address, offset_size) >>
        address_of_file_free_space_info: call!(address, offset_size) >>
        end_of_file_address: call!(address, offset_size) >>
        driver_information_block_address: call!(address, offset_size) >>
        root_group_symbol_table_entry: call!(parse_symbol_table_entry, offset_size) >>
        (Hdf5Superblock {
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
        })
    )
);

#[derive(Debug)]
pub struct SymbolTable {
    pub version: u8,
    pub entries: Vec<SymbolTableEntry>,
}

named_args!(pub parse_symbol_table (offset_size: u8) <SymbolTable>,
    do_parse!(
        tag!(b"SNOD") >>
        version: le_u8 >>
        tag!(b"\0") >>
        number_of_symbols: le_u16 >>
        entries: count!(call!(parse_symbol_table_entry, offset_size), number_of_symbols as usize) >>
        (SymbolTable {
            version,
            entries,
        })
    )
);

#[derive(Debug)]
pub struct SymbolTableEntry {
    pub link_name_offset: u64,
    pub object_header_address: u64,
    pub cache_type: u32,
    pub address_of_btree: u64,
    pub address_of_name_heap: u64,
}

named_args!(pub parse_symbol_table_entry (offset_size: u8) <SymbolTableEntry>,
    do_parse!(
        link_name_offset: call!(address, offset_size) >>
        object_header_address: call!(address, offset_size) >>
        cache_type: le_u32 >>
        tag!(b"\0\0\0\0") >>
        address_of_btree: call!(address, offset_size) >>
        address_of_name_heap: call!(address, offset_size) >>
        (SymbolTableEntry {
            link_name_offset,
            object_header_address,
            cache_type,
            address_of_btree,
            address_of_name_heap,
        })
    )
);

#[derive(Debug, Clone)]
pub struct GroupEntry {
    pub byte_offset_into_local_heap: u64,
    pub pointer_to_symbol_table: u64,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(pub parse_group_entry<GroupEntry>,
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
        node: call!(parse_group_node, offset_size) >>
        (node)
    )
);

named_args!(parse_group_node (offset_size: u8) <GroupNode>,
    do_parse!(
        node_level: le_u8 >>
        entries_used: le_u16 >>
        address_of_left_sibling: call!(address, offset_size) >>
        address_of_right_sibling: call!(address, offset_size) >>
        entries: count!(parse_group_entry, entries_used as usize) >>
        (GroupNode {
            node_level,
            entries_used,
            address_of_left_sibling,
            address_of_right_sibling,
            entries,
        })
    )
);

#[derive(Debug)]
pub struct LocalHeap {
    pub version: u8,
    pub data_segment_size: u64,
    pub offset_to_head_of_freelist: u64,
    pub address_of_data_segment: u64,
}

named_args!(pub parse_local_heap (offset_size: u8, length_size: u8) <LocalHeap>,
    do_parse!(
        dbg!(tag!(b"HEAP")) >>
        version: le_u8 >>
        tag!(b"\0\0\0") >>
        data_segment_size: call!(address, length_size) >>
        offset_to_head_of_freelist: call!(address, length_size) >>
        address_of_data_segment: call!(address, offset_size) >>
        (LocalHeap {
            version,
            data_segment_size,
            offset_to_head_of_freelist,
            address_of_data_segment,
        })
    )
);

#[derive(Debug)]
pub struct ObjectHeader {
    pub version: u8,
    pub total_number_of_header_messages: u16,
    pub object_reference_count: u32,
    pub object_header_size: u32,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(pub parse_object_header <ObjectHeader>,
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
    pub struct DataType {
        pub version: u8,
        pub class: u8,
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

#[cfg_attr(rustfmt, rustfmt_skip)]
named_args!(parse_datatype (message_size: u16) <header::DataType>,
    do_parse!(
        class_and_version: le_u8 >>
        class_bitfields: le_u24 >>
        size: le_u32 >>
        properties: count!(le_u8, message_size as usize - 8) >>
        (header::DataType {
            version: class_and_version >> 4,
            class: class_and_version & 0b00001111,
            class_bitfields,
            size,
            properties,
        })
    )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_dataspace <header::Dataspace>,
    do_parse!(
        version: le_u8 >>
        dimensionality: le_u8 >>
        flags: le_u8 >>
        take!(5) >> // not reqired to be zero, oddly
        dimensions: count!(apply!(address, 8), dimensionality as usize) >>
        _max_dimensions: cond!(flags == 1, count!(apply!(address, 8), dimensionality as usize)) >>
        (header::Dataspace {
            version,
            dimensionality,
            flags,
            dimensions,
        })
    )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_fill_value <header::DataStorageFillValue>,
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

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_data_layout <header::DataLayout>,
    do_parse!(
        dbg!(tag!([3u8])) >> // version 3, only implement version 3 because I'm lazy
        dbg!(tag!([1u8])) >> // layout class 1, contiguious storage
        data_address: call!(address, 8) >>
        size: call!(address, 8) >>
        take!(6) >> // TODO: pad to a multiple of 8 bytes
        (header::DataLayout {
            address: data_address,
            size,
        })
    )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named_args!(parse_attribute (message_size: u16) <header::Attribute>,
    do_parse!(
        dbg!(tag!([1])) >> // we only handle version 1
        dbg!(tag!([0])) >>
        name_size: le_u16 >>
        datatype_size: le_u16 >>
        dataspace_size: le_u16 >>
        name: take!(pad8(name_size)) >>
        datatype: call!(parse_datatype, datatype_size) >>
        dataspace: parse_dataspace >>
        data: take!(message_size as usize - (8 + pad8(name_size) + pad8(datatype_size) + pad8(dataspace_size))) >>
        (header::Attribute {
            name: name.iter().take_while(|b| **b > 0).map(|b| *b as char).collect::<String>(),
            datatype,
            dataspace,
            data: data.to_vec(),
        })
    )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_object_header_continuation <header::ObjectHeaderContinuation>,
    do_parse!(
        offset: call!(address, 8) >>
        length: call!(address, 8) >>
        (header::ObjectHeaderContinuation {
            length,
            offset,
        })
    )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_symbol_table_message <header::SymbolTable>,
    do_parse!(
        btree_address: call!(address, 8) >>
        local_heap_address: call!(address, 8) >>
        (header::SymbolTable {
            btree_address,
            local_heap_address,
        })
    )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_object_modification_time <header::ObjectModificationTime>,
    do_parse!(
        dbg!(tag!([1])) >> // version 1 is the only one allowed by the standard
        dbg!(tag!(b"\0\0\0")) >>
        seconds: le_u32 >>
        (header::ObjectModificationTime {
            seconds_after_unix_epoch: seconds,
        })
    )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(pub parse_header_message <header::Message>,
    do_parse!(
        message_type: le_u16 >>
        message_size: le_u16 >>
        _flags: le_u8 >>
        dbg!(tag!(b"\0\0\0")) >>
        message: switch!(value!(message_type),
            0x0 => value!(header::Message::Nil) |
            0x1 => map!(call!(parse_dataspace), header::Message::Dataspace) |
            0x3 => map!(call!(parse_datatype, message_size), header::Message::DataType) |
            0x5 => map!(call!(parse_fill_value), header::Message::DataStorageFillValue) |
            0x8 => map!(call!(parse_data_layout), header::Message::DataLayout) |
            0xC => map!(call!(parse_attribute, message_size), header::Message::Attribute) |
            0x10 => map!(call!(parse_object_header_continuation), header::Message::ObjectHeaderContinuation) |
            0x11 => map!(call!(parse_symbol_table_message), header::Message::SymbolTable) |
            0x12 => map!(call!(parse_object_modification_time), header::Message::ObjectModificationTime) |
            _ => value!(header::Message::Nil)
        ) >>
        (message)
    )
);

/*
fn padding_for<T>(t: T) -> usize
where
    usize: From<T>,
{
    let t = usize::from(t);
    if t % 8 == 0 {
        0
    } else {
        8 - (t % 8)
    }
}

#[test]
fn test_padding() {
    assert!(padding_for(0u8) == 0);
    assert!(padding_for(1u8) == 7);
    assert!(padding_for(2u8) == 6);
    assert!(padding_for(3u8) == 5);
    assert!(padding_for(4u8) == 4);
    assert!(padding_for(5u8) == 3);
    assert!(padding_for(6u8) == 2);
    assert!(padding_for(7u8) == 1);
    assert!(padding_for(8u8) == 0);
}
*/

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

// This assumes version 0
/*
fn main() -> Result<(), crate::Error> {
    let file = std::fs::read("test.hdf5")?;

    // Read the superblock
    let superblock = parse_superblock(&file)?.1;
    println!("{:#?}", superblock);

    // Read the local heap
    let heap = parse_local_heap(
        &file[superblock
            .root_group_symbol_table_entry
            .address_of_name_heap as usize..],
        superblock.offset_size,
        superblock.length_size,
    )?
    .1;
    println!("{:#?}", heap);

    let header = parse_object_header(
        &file[superblock
            .root_group_symbol_table_entry
            .object_header_address as usize..],
        1,
    )?
    .1;
    println!("{:#?}", header);

    if let HeaderMessage::ObjectHeaderContinuation { offset, .. } = header.messages[0] {
        let message = parse_header_message(&file[offset as usize..])?.1;
        println!("{:#?}", message);
        if let HeaderMessage::SymbolTable { btree_address, .. } = message {
            let node = hdf5_node(&file[btree_address as usize..], superblock.offset_size)?.1;
            println!("{:#?}", node);
            if let BtreeNode::GroupNode { entries, .. } = node {
                let table = parse_symbol_table(
                    &file[entries[0].pointer_to_symbol_table as usize..],
                    superblock.offset_size,
                )?
                .1;
                println!("{:#?}", table);
                let name = &file
                    [(table.entries[0].link_name_offset + heap.address_of_data_segment) as usize..]
                    .iter()
                    .take_while(|b| **b != 0)
                    .map(|b| *b as char)
                    .collect::<String>();
                println!("name is: {}", name);
                let header = parse_object_header(
                    &file[table.entries[0].object_header_address as usize..],
                    6,
                )?
                .1;
                println!("{:#?}", header);
            }
        }
    }

    return Ok(());

    // Now having read the superblock and heap we can start traversing the Btree structure

    // Read the btree root node
    let root_node = hdf5_node(
        &file[superblock.root_group_symbol_table_entry.address_of_btree as usize..],
        superblock.offset_size,
    )?
    .1;
    println!("{:#?}", root_node);

    // If we parsed a group node, read the next node down
    if let BtreeNode::GroupNode { entries, .. } = root_node {
        for entry in entries {
            let table = parse_symbol_table(
                &file[entry.pointer_to_symbol_table as usize..],
                superblock.offset_size,
            )?
            .1;
            println!("{:#?}", table);
            // Read the name of the symbol table entry out of the local heap
            let name = &file
                [(table.entries[0].link_name_offset + heap.address_of_data_segment) as usize..]
                .iter()
                .take_while(|b| **b != 0)
                .map(|b| *b as char)
                .collect::<String>();
            println!("name is: {}", name);

            // TODO: need to leave breadcrumbs for myself next time...
            // What's going on here?

            // Read the object header
            let header =
                parse_object_header(&file[table.entries[0].object_header_address as usize..], 1)?.1;
            println!("{:#?}", header);
        }
    }

    Ok(())
}
*/
