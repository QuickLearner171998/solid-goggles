# Wedding Album Auto-Selector for Indian Weddings

An intelligent wedding photo selection system specifically designed for Indian marriages, using local quality metrics and LLM vision scoring to automatically curate the best images from your wedding collection with cultural understanding.

## Features

- **Indian Wedding Expertise**: Specialized prompts and scoring for Indian wedding ceremonies (Haldi, Mehendi, Sangeet, Pheras, Varmala, Vidaai, etc.)
- **Ceremony-Aware Curation**: Ensures balanced representation across different rituals and moments
- **Cultural Detail Recognition**: Identifies and prioritizes shots of mehendi designs, jewelry, traditional attire, and cultural elements
- **Emotional Moment Detection**: Specifically tuned to capture key emotional moments like Vidaai, family embraces, and candid joy
- **Local Quality Assessment**: Evaluates sharpness, exposure, colorfulness, and face detection
- **Perceptual Deduplication**: Removes near-duplicate images using perceptual hashing
- **Advanced LLM Vision Scoring**: Uses OpenAI's GPT-4.1 with expert Indian wedding photography prompts
- **Modular Architecture**: Clean OOP design with pluggable components
- **Batch Processing**: Efficient memory management with configurable batch sizes
- **Distribution Reports**: Shows how images are distributed across ceremony types

## Architecture

The system is built with modular, reusable components:

```
wedding_album_selector/
├── config.py                    # Central configuration
├── wedding_album_selector.py    # Main orchestrator
├── run.py                       # Entry point
├── components/
│   ├── __init__.py
│   ├── image_loader.py          # Image loading from filesystem
│   ├── image_processor.py       # Quality metrics and image processing
│   ├── llm_scorer.py            # LLM-based scoring with Indian wedding expertise
│   ├── deduplicator.py          # Near-duplicate detection
│   ├── selector.py              # Selection and ranking
│   └── album_curator.py         # Balanced ceremony distribution (NEW!)
├── test/
│   └── test_components.py       # Component tests
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Or export it as an environment variable:
```bash
export OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Place your wedding photos in the `album_pics/` directory (or configure a different source in `config.py`)

2. Run the selector:
```bash
python run.py
```

3. Find results in `album_selection_top1000/`:
   - Selected images (top 1000 by default)
   - `album_scores_log.csv` - detailed scoring report

## Configuration

Edit `config.py` to customize:

- `SOURCE_FOLDER`: Path to your wedding photos
- `OUTPUT_DIR`: Output directory for selected images
- `NUM_SELECT`: Number of images to select (default: 1000)
- `BATCH_SIZE`: Processing batch size (default: 5)
- `LLM_PREFILTER_TARGET`: Number of images to send to LLM (default: 1800)
- `OPENAI_VISION_MODEL`: LLM model to use (default: gpt-4.1)
- `USE_BALANCED_CURATION`: Enable ceremony-balanced selection (default: True)

## How It Works

1. **Discovery**: Scans source folder for all supported images (.jpg, .jpeg, .png)

2. **Local Prefiltering**: 
   - Calculates quality metrics (sharpness, exposure, colorfulness)
   - Detects faces using Haar cascades
   - Removes near-duplicates using perceptual hashing
   - Keeps best image per hash

3. **LLM Scoring with Indian Wedding Expertise**:
   - Creates thumbnails for efficient processing
   - Sends batches to GPT-4.1 vision model with specialized prompts
   - Scores on:
     - **Technical Quality**: Sharpness, exposure, color accuracy
     - **Composition**: Framing, rule of thirds, creative angles
     - **Moment Value**: Identifies key ceremonies (Pheras, Varmala, Haldi, Mehendi, Sangeet, Vidaai)
     - **Cultural Details**: Recognizes jewelry, mehendi patterns, traditional attire
     - **Emotional Impact**: Detects genuine emotions, tears, joy, candid moments
   - Tags images by ceremony type, subject, and moment

4. **Intelligent Curation**:
   - Ensures balanced distribution across ceremony types:
     - 35% Key Rituals (Pheras, Varmala, Sindoor, Mangalsutra)
     - 20% Emotional Moments (Vidaai, candid emotions)
     - 15% Pre-Wedding (Haldi, Mehendi, Sangeet)
     - 10% Details (Jewelry, decor, venue)
     - 12% Portraits (Couple, family)
     - 8% Other (Baraat, reception, misc)
   - Combines local (25%) and LLM (75%) scores
   - Selects top N images while maintaining ceremony diversity

5. **Reporting**:
   - Generates detailed CSV with all scores and metrics
   - Shows distribution analysis across ceremony types
   - Preserves original filenames with collision handling

## Component Details

### ImageLoader
Handles file system operations and image discovery. Pluggable - can be replaced with cloud storage loaders.

### ImageProcessor
Calculates local quality metrics:
- Sharpness (Laplacian variance)
- Exposure (histogram analysis)
- Colorfulness (color space statistics)
- Face detection (Haar cascades)

### Deduplicator
Uses perceptual hashing to identify and remove near-duplicates, keeping the highest quality version.

### LLMScorer
Interfaces with OpenAI's vision models using expert-level prompts for Indian wedding photography. Recognizes and prioritizes:
- Traditional ceremonies (Haldi, Mehendi, Sangeet, Pheras, Varmala, etc.)
- Cultural details (mehendi designs, jewelry, traditional attire)
- Emotional moments (Vidaai tears, family hugs, candid joy)
- Includes automatic retry logic and detailed tagging

### AlbumCurator (NEW!)
Ensures balanced representation of different ceremony types in the final album. Prevents over-representation of any single ceremony and ensures all key moments are captured.

### Selector
Handles final ranking, selection, and output operations including CSV generation and file copying.

## Customization

The modular design allows easy customization:

- Replace `ImageLoader` to load from cloud storage
- Modify `ImageProcessor` to add custom quality metrics
- Swap `LLMScorer` with different AI models
- Extend `Selector` with custom ranking algorithms

## License

MIT License - feel free to use and modify as needed.
