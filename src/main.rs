use nom::*;

#[derive(Debug)]
struct Hdf5Superblock {
    superblock_version: u8,
    free_space_storage_version: u8,
    root_group_symbol_table_entry_version: u8,
    // reserved 0 byte
    shared_header_message_format_version: u8,
    offset_size: u8,
    length_size: u8,
    // reserved 0 byte
    group_leaf_node_k: u16,
    group_internal_node_k: u16,
    file_consistency_flags: u32,
    base_address: u64,
    address_of_file_free_space_info: u64,
    end_of_file_address: u64,
    driver_information_block_address: u64,
    root_group_symbol_table_entry: SymbolTableEntry,
}

named_args!(address (len: u8) <u64>,
   map!(take!(len), |x| le_u64(x).unwrap().1)
);

named!(parse_superblock<&[u8], Hdf5Superblock>,
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
        root_group_symbol_table_entry: call!(hdf5_symbol_table_entry, offset_size) >>
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
struct SymbolTable {
    version: u8,
    entries: Vec<SymbolTableEntry>,
}

named_args!(parse_symbol_table (offset_size: u8) <SymbolTable>,
    do_parse!(
        tag!(b"SNOD") >>
        version: le_u8 >>
        tag!(b"\0") >>
        number_of_symbols: le_u16 >>
        entries: count!(call!(hdf5_symbol_table_entry, offset_size), number_of_symbols as usize) >>
        (SymbolTable {
            version,
            entries,
        })
    )
);

#[derive(Debug)]
struct SymbolTableEntry {
    link_name_offset: u64,
    object_header_address: u64,
    cache_type: u32,
    address_of_btree: u64,
    address_of_name_heap: u64,
}

named_args!(hdf5_symbol_table_entry (offset_size: u8) <SymbolTableEntry>,
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

#[derive(Debug)]
struct Key {
    chunk_size: u32,
    filter_mask: u32,
    axis_offsets: Vec<u64>,
}

named!(
    parse_key<Key>,
    do_parse!(
        chunk_size: le_u32
            >> filter_mask: le_u32
            >> axis_offsets: read_until_zero
            >> (Key {
                chunk_size,
                filter_mask,
                axis_offsets,
            })
    )
);

fn read_until_zero(remaining: &[u8]) -> nom::IResult<&[u8], Vec<u64>> {
    let mut output = Vec::new();
    loop {
        match le_u64(remaining) {
            Ok((remaining, val)) => {
                if val == 0 {
                    return Ok((remaining, output));
                }
                output.push(val);
            }
            Err(e) => return Err(e),
        }
    }
}

#[derive(Debug)]
struct GroupEntry {
    byte_offset_into_local_heap: u64,
    pointer_to_symbol_table: u64,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_group_entry<GroupEntry>,
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
struct DataChunkEntry {
    key: Key,
    pointer_to_data_chunk: u64,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_data_chunk_entry<DataChunkEntry>,
    do_parse!(
        key: parse_key >>
        pointer_to_data_chunk: le_u64 >>
        (DataChunkEntry {
            key,
            pointer_to_data_chunk,
        })
    )
);

#[derive(Debug)]
enum BtreeNode {
    DataChunkNode {
        node_level: u8,
        entries_used: u16,
        address_of_left_sibling: u64,
        address_of_right_sibling: u64,
        entries: Vec<DataChunkEntry>,
    },
    GroupNode {
        node_level: u8,
        entries_used: u16,
        address_of_left_sibling: u64,
        address_of_right_sibling: u64,
        entries: Vec<GroupEntry>,
    },
}

named_args!(hdf5_node (offset_size: u8) <BtreeNode>,
    do_parse!(
        tag!(b"TREE") >>
        node: switch!( le_u8,
            0 => call!(parse_group_node, offset_size) |
            1 => call!(parse_data_chunk_node, offset_size)
        ) >>
        (node)
    )
);

/*
named_args!(hdf5_node (offset_size: u8) <BtreeNode>,
    do_parse!(
        tag!(b"TREE") >>
        node_type: le_u8 >>
        node_level: le_u8 >>
        entries_used: le_u16 >>
        address_of_left_sibling: call!(address, offset_size) >>
        address_of_right_sibling: call!(address, offset_size) >>
    )
);
*/

named_args!(parse_group_node (offset_size: u8) <BtreeNode>,
    do_parse!(
        node_level: le_u8 >>
        entries_used: le_u16 >>
        address_of_left_sibling: call!(address, offset_size) >>
        address_of_right_sibling: call!(address, offset_size) >>
        entries: count!(parse_group_entry, entries_used as usize) >>
        (BtreeNode::GroupNode {
            node_level,
            entries_used,
            address_of_left_sibling,
            address_of_right_sibling,
            entries,
        })
    )
);

named_args!(parse_data_chunk_node (offset_size: u8) <BtreeNode>,
    do_parse!(
        node_level: le_u8 >>
        entries_used: le_u16 >>
        address_of_left_sibling: call!(address, offset_size) >>
        address_of_right_sibling: call!(address, offset_size) >>
        entries: count!(parse_data_chunk_entry, entries_used as usize) >>
        (BtreeNode::DataChunkNode {
            node_level,
            entries_used,
            address_of_left_sibling,
            address_of_right_sibling,
            entries,
        })
    )
);

#[derive(Debug)]
struct LocalHeap {
    version: u8,
    data_segment_size: u64,
    offset_to_head_of_freelist: u64,
    address_of_data_segment: u64,
}

named_args!(parse_local_heap (offset_size: u8, length_size: u8) <LocalHeap>,
    do_parse!(
        tag!(b"HEAP") >>
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
struct ObjectHeader {
    version: u8,
    total_number_of_header_messages: u16,
    object_reference_count: u32,
    object_header_size: u32,
    //message: HeaderMessageType,
    messages: Vec<HeaderMessageType>,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_object_header <ObjectHeader>,
    do_parse!(
        version: le_u8 >>
        tag!(b"\0") >>
        total_number_of_header_messages: le_u16 >>
        object_reference_count: le_u32 >>
        object_header_size: le_u32 >>
        //message: parse_header_message >>
        messages: count!(call!(parse_header_message), total_number_of_header_messages as usize) >>
        (ObjectHeader {
            version,
            total_number_of_header_messages,
            object_reference_count,
            object_header_size,
            messages,
        })

    )
);

#[derive(Debug)]
enum HeaderMessageType {
    Nil,
    Dataspace,
    LinkInfo,
    DataType,
    DataStorageFillValue,
    Link,
    DataStorageExternal,
    DataLayout,
    Bogus,
    GroupInfo,
    DataStorageFilterPipeline,
    Attribute,
    ObjectComment,
    SharedMessageTable,
    ObjectHeaderContinuation,
    SymbolTable,
    ObjectModificationTime,
    BtreeKValues,
    DriverInfo,
    AttributeInfo,
    ObjectReferenceCount,
}

use HeaderMessageType::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_header_message <HeaderMessageType>,
    do_parse!(
        message_type: le_u16 >>
        message_size: le_u16 >>
        flags: le_u8 >>
        //tag!(b"\0\0\0") >>
        take!(3) >>
        take!(message_size) >>
        ({
            println!("{:08b}", flags);
            match message_type {
            // These are hex because the docs use hex
            0x0 => Nil,
            0x1 => Dataspace,
            0x2 => LinkInfo,
            0x3 => DataType,
            0x5 => DataStorageFillValue,
            0x6 => Link,
            0x7 => DataStorageExternal,
            0x8 => DataLayout,
            0x9 => Bogus,
            0xA => GroupInfo,
            0xB => DataStorageFilterPipeline,
            0xC => Attribute,
            0xD => ObjectComment,
            0xF => SharedMessageTable,
            0x10 => ObjectHeaderContinuation,
            0x11 => SymbolTable,
            0x12 => ObjectModificationTime,
            0x13 => BtreeKValues,
            0x14 => DriverInfo,
            0x15 => AttributeInfo,
            0x16 => ObjectReferenceCount,
            _ => panic!("Invalid header message type"),

        }})
    )
);

// This assumes version 0
fn main() {
    let file = std::fs::read("test.hdf5").unwrap();

    // Read the superblock
    let superblock = parse_superblock(&file).unwrap().1;
    println!("{:#?}", superblock);

    // Read the local heap
    let heap = parse_local_heap(
        &file[superblock
            .root_group_symbol_table_entry
            .address_of_name_heap as usize..],
        superblock.offset_size,
        superblock.length_size,
    )
    .unwrap()
    .1;
    println!("{:#?}", heap);

    let name = &file[(superblock.root_group_symbol_table_entry.link_name_offset
        + heap.address_of_data_segment) as usize..]
        .iter()
        .take_while(|b| **b != 0)
        .map(|b| *b as char)
        .collect::<String>();
    println!("name is: {:?}", name);

    let header = parse_object_header(
        &file[superblock
            .root_group_symbol_table_entry
            .object_header_address as usize..],
    )
    .unwrap()
    .1;
    println!("{:#?}", header);

    // Now having read the superblock and heap we can start traversing the Btree structure

    // Read the btree root node
    let root_node = hdf5_node(
        &file[superblock.root_group_symbol_table_entry.address_of_btree as usize..],
        superblock.offset_size,
    )
    .unwrap()
    .1;
    println!("{:#?}", root_node);

    // If we parsed a group node, read the next node down
    if let BtreeNode::GroupNode { entries, .. } = root_node {
        let table = parse_symbol_table(
            &file[entries[0].pointer_to_symbol_table as usize..],
            superblock.offset_size,
        )
        .unwrap()
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

        // Read the object header
        let header = parse_object_header(&file[table.entries[0].object_header_address as usize..])
            .unwrap()
            .1;
        println!("{:#?}", header);
    }
}
