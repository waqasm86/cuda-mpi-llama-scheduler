# Documentation Index

Welcome to the cuda-mpi-llama-scheduler documentation. This directory contains detailed technical documentation, guides, and analysis reports.

## Core Documentation

### [ARCHITECTURE.md](ARCHITECTURE.md)
Detailed system architecture documentation covering:
- MPI scheduler design (rank-based distribution)
- llama.cpp server integration
- CUDA post-processing kernel implementation
- Data flow and metrics collection
- Design patterns and scalability analysis

### [PROJECT_STATUS.md](PROJECT_STATUS.md)
Comprehensive project status report including:
- System dependencies verification
- Build and compilation status
- Integration testing results
- Performance benchmarks
- Bash scripts functionality
- Next steps and recommendations

### [GPU_USAGE_GUIDE.md](GPU_USAGE_GUIDE.md)
GPU usage verification and optimization guide:
- How to verify GPU is being used
- VRAM allocation breakdown
- Performance comparison (GPU vs CPU)
- Optimization recommendations
- Troubleshooting GPU issues
- Command reference for monitoring

### [LOG_ANALYSIS.md](LOG_ANALYSIS.md)
Detailed log analysis and performance profiling:
- Scheduler execution logs analysis
- llama.cpp server performance breakdown
- Request processing timeline
- Prompt cache efficiency metrics
- Optimization recommendations
- Health status summary

## Quick Links

**Getting Started**: See [../QUICKSTART.md](../QUICKSTART.md) for step-by-step setup

**Recent Changes**: See [../CHANGELOG.md](../CHANGELOG.md) for latest updates

**Main Project**: See [../README.md](../README.md) for project overview

## Documentation Organization

```
cuda-mpi-llama-scheduler/
├── README.md              # Project overview and features
├── QUICKSTART.md          # Quick start guide
├── CHANGELOG.md           # Recent changes and improvements
└── docs/
    ├── README.md          # This file - documentation index
    ├── ARCHITECTURE.md    # System architecture details
    ├── PROJECT_STATUS.md  # Project status report
    ├── GPU_USAGE_GUIDE.md # GPU verification and optimization
    └── LOG_ANALYSIS.md    # Performance analysis
```

## Reading Order Recommendations

### For New Users
1. [../README.md](../README.md) - Understand what the project does
2. [../QUICKSTART.md](../QUICKSTART.md) - Get it running on your system
3. [GPU_USAGE_GUIDE.md](GPU_USAGE_GUIDE.md) - Verify GPU is working correctly

### For Developers
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the system design
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) - See what's implemented and tested
3. [LOG_ANALYSIS.md](LOG_ANALYSIS.md) - Learn about performance characteristics

### For Performance Tuning
1. [GPU_USAGE_GUIDE.md](GPU_USAGE_GUIDE.md) - Optimize GPU layer offloading
2. [LOG_ANALYSIS.md](LOG_ANALYSIS.md) - Analyze bottlenecks and metrics
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand inflight concurrency patterns

## Key Topics by Document

### System Setup
- **QUICKSTART.md**: Prerequisites, build, testing
- **PROJECT_STATUS.md**: Dependencies verification, integration testing

### GPU Usage
- **GPU_USAGE_GUIDE.md**: VRAM monitoring, layer offloading, optimization
- **LOG_ANALYSIS.md**: GPU utilization analysis, performance metrics

### Architecture
- **ARCHITECTURE.md**: MPI design, worker pool pattern, metrics aggregation
- **PROJECT_STATUS.md**: Component breakdown, data flow

### Performance
- **LOG_ANALYSIS.md**: Throughput, latency, cache efficiency
- **GPU_USAGE_GUIDE.md**: Performance comparison, tuning recommendations

## Contributing to Documentation

When adding new documentation:
1. Place technical details in `docs/`
2. Keep user-facing guides at root level
3. Update this index with new files
4. Add navigation links in main README.md

## Documentation Standards

- Use markdown format (.md)
- Include code examples with syntax highlighting
- Add diagrams for complex concepts (ASCII art is fine)
- Reference specific file paths with line numbers when relevant
- Include command examples with expected output
- Keep language clear and concise

## Support

For issues or questions:
- Check [../QUICKSTART.md](../QUICKSTART.md) troubleshooting section
- Review [GPU_USAGE_GUIDE.md](GPU_USAGE_GUIDE.md) for GPU issues
- See [PROJECT_STATUS.md](PROJECT_STATUS.md) for known issues

## Version History

See [../CHANGELOG.md](../CHANGELOG.md) for detailed version history and recent changes.
