fn main() {
    let file = hdf5::read("tot_SMC.h5").unwrap();
    println!("L_Lya: {:.4e}", file.attr::<f64>("L_Lya"));
    println!("freq_type: {:?}", file.attr::<String>("LOS/SB/units"));
}
