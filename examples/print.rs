fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = std::env::args().nth(1).unwrap_or(String::from("test.hdf5"));
    let file = hdf5::Hdf5File::open(filename)?;
    let temperature = file.view::<f64>("T");
    let child_check = file.view::<i32>("child_check");
    println!("{:#?}", file);

    let total_temp = child_check
        .iter()
        .zip(temperature.iter())
        .filter_map(|(c, t)| if *c != 1 { Some(t) } else { None })
        .sum::<f64>();
    println!("{}", total_temp);
    Ok(())
}
