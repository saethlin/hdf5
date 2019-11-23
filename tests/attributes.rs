extern crate hdf5;

static PROGRAM: &'static str = "\
import numpy as np
import h5py
with h5py.File('attributes.hdf5', 'w') as f:
    f.attrs['i32_attribute'] = np.int32(12345)
    f.attrs['i64_attribute'] = np.int64(12345)
    f.attrs['f32_attribute'] = np.float32(1.2345)
    f.attrs['f64_attribute'] = np.float64(1.2345)
";

#[test]
fn can_parse_attribute() {
    let status = std::process::Command::new("python3.8")
        .arg("-c")
        .arg(PROGRAM)
        .status()
        .expect("Unable to generate the file");
    assert!(status.success());

    let file = hdf5::read("attributes.hdf5").expect("Unable to open the file");
    println!("{:#?}", file);

    assert_eq!(file.attr::<i32>("i32_attribute"), 12345);
    assert_eq!(file.attr::<i64>("i64_attribute"), 12345);
    assert_eq!(file.attr::<f32>("f32_attribute"), 1.2345);
    assert_eq!(file.attr::<f64>("f64_attribute"), 1.2345);
}
