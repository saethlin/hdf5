#[derive(Debug)]
pub enum Hdf5Error {
    Io(std::io::Error),
    Parse(String),
}

impl From<std::io::Error> for Hdf5Error {
    fn from(e: std::io::Error) -> Self {
        Hdf5Error::Io(e)
    }
}

impl From<nom::Err<&[u8]>> for Hdf5Error {
    fn from(e: nom::Err<&[u8]>) -> Self {
        use nom::Context;
        use nom::Err::*;
        match &e {
            Incomplete(_) => Hdf5Error::Parse(format!("Incomplete: {:?}", e)),
            Error(Context::Code(_, e)) => Hdf5Error::Parse(format!("Error: {:?}", e)),
            Failure(Context::Code(_, e)) => Hdf5Error::Parse(format!("Failure: {:?}", e)),
        }
    }
}
