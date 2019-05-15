/// All errors that this library can emit
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{}", e),
            Error::Parse(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<nom::Err<&[u8]>> for Error {
    fn from(e: nom::Err<&[u8]>) -> Self {
        use nom::Context;
        match &e {
            nom::Err::Incomplete(_) => Error::Parse(format!("Incomplete: {:?}", e)),
            nom::Err::Error(Context::Code(_, e)) => Error::Parse(format!("Error: {:?}", e)),
            nom::Err::Failure(Context::Code(_, e)) => Error::Parse(format!("Failure: {:?}", e)),
        }
    }
}
