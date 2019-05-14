fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = std::env::args().nth(1).unwrap_or(String::from("test.hdf5"));
    let file = hdf5::open(filename)?;
    println!("{:#?}", file);
    Ok(())
}
