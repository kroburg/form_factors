#define EPS 0.000001

#ifdef _DEBUG
#define TRACE(msg) \
    std::cout << "TRACE " << __DATE__ << __TIME__ << " " << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl
#else
#define TRACE(msg)
#endif

#define LOG(msg) \
    std::cout << "INFO " << __DATE__ << __TIME__ << " " << msg << std::endl

#define ERROR(msg) \
    std::cerr << "ERROR " << __DATE__ << __TIME__ << " " << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl