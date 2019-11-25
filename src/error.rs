/// All errors that this library can emit
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

impl std::fmt::Debug for Error {
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

impl From<nom::Err<nom::error::VerboseError<&[u8]>>> for Error {
    fn from(e: nom::Err<nom::error::VerboseError<&[u8]>>) -> Self {
        match e {
            nom::Err::Incomplete(needed) => Error::Parse(format!("{:?}", needed)),
            nom::Err::Error(e) | nom::Err::Failure(e) => {
                let mut trace = String::from("\n");
                for (_, reason) in e.errors {
                    trace.extend(format!("    {:?}\n", reason).chars());
                }
                trace.pop();
                Error::Parse(trace)
            }
        }
    }
}
