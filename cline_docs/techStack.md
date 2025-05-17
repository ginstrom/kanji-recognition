## Backend
- Language: Python
- Machine Learning: PyTorch
- Image Processing: OpenCV, PIL
- Data Storage: LMDB
- Font Processing: fonttools, matplotlib.font_manager

## DevOps
- Containerization: Docker
- Execution Environment: Docker Compose
- Build Automation: Makefile

## Development Workflow
- All Python scripts should be executed within the Docker Compose environment to ensure consistency and manage dependencies.
- Use the Makefile for common operations:
  - `make run CMD` - Run a command in the Docker container
  - `make shell` - Open a shell in the Docker container
  - `make build` - Build the Docker container
  - `make test` - Run unit tests in the Docker container
  - `make clean` - Delete all images from output/prep directory
  - `make help` - Display help information

## Image Processing Technologies
- OpenCV for advanced image processing operations:
  - Multiple thresholding methods (Otsu, adaptive, fixed)
  - Morphological operations for stroke connectivity
  - Gaussian blur for noise reduction
- PIL (Python Imaging Library) for basic image operations:
  - Font rendering for synthetic kanji images
  - Text positioning and drawing
  - Image format conversion
- Custom algorithms for touch/stylus input optimization
- Font detection and rendering for synthetic data generation

## Machine Learning Technologies
- PyTorch for model development and training
- CNN architecture for kanji recognition
- Data augmentation for touch input simulation
- Mixed precision training for performance optimization
- Synthetic data generation for improved model generalization

## Data Generation
- ETL9G dataset for handwritten kanji samples
- Font-based kanji image generation for additional training data
- System font detection and filtering for Japanese support
- Multiple font styles (regular, bold) for style variation
