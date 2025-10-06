# Wedding Album Auto-Selector for Indian Weddings (Enhanced)

An intelligent wedding photo selection system specifically designed for Indian marriages, using advanced AI techniques including CLIP embeddings, clustering, person detection, and LLM vision scoring to automatically curate the best images from your wedding collection with deep cultural understanding.

## 🚀 New Enhanced Features

- **CLIP Image Embeddings**: State-of-the-art image embeddings using CLIP (ViT-B-32) for semantic understanding
- **Smart Clustering**: Automatic grouping of similar images using K-Means/DBSCAN with UMAP dimensionality reduction
- **Person Detection & Categorization**: AI-powered identification of bride, groom, and family members
- **Person-Specific Organization**: Automatically organizes selected photos into directories:
  - Bride photos
  - Groom photos
  - Couple photos
  - Bride's parents (mom, dad, both)
  - Bride's siblings
  - Groom's parents (mom, dad, both)
  - Family groups
  - Ceremony-focused shots
  - Guests and children
- **Cluster-Based Selection**: Selects best representatives from each cluster to ensure diversity
- **Verbose Logging**: Saves intermediate results at each pipeline stage for analysis
- **Enhanced Pipeline**: 7-phase intelligent processing with comprehensive statistics

## Core Features

- **Indian Wedding Expertise**: Specialized prompts and scoring for Indian wedding ceremonies (Haldi, Mehendi, Sangeet, Pheras, Varmala, Vidaai, etc.)
- **Ceremony-Aware Curation**: Ensures balanced representation across different rituals and moments
- **Cultural Detail Recognition**: Identifies and prioritizes shots of mehendi designs, jewelry, traditional attire, and cultural elements
- **Emotional Moment Detection**: Specifically tuned to capture key emotional moments like Vidaai, family embraces, and candid joy
- **Local Quality Assessment**: Evaluates sharpness, exposure, colorfulness, and face detection
- **Perceptual Deduplication**: Removes near-duplicate images using perceptual hashing
- **Advanced LLM Vision Scoring**: Uses OpenAI's GPT-4o with expert Indian wedding photography prompts
- **Modular Architecture**: Clean OOP design with pluggable components
- **Batch Processing**: Efficient memory management with configurable batch sizes
- **Distribution Reports**: Shows how images are distributed across ceremony types and people

## Architecture

The system is built with modular, reusable components:

```
wedding_album_selector/
├── config.py                    # Central configuration with enhanced features
├── wedding_album_selector.py    # Main orchestrator (enhanced 7-phase pipeline)
├── run.py                       # Entry point
├── components/
│   ├── __init__.py
│   ├── image_loader.py          # Image loading from filesystem
│   ├── image_processor.py       # Quality metrics and image processing
│   ├── llm_scorer.py            # LLM-based scoring with Indian wedding expertise
│   ├── deduplicator.py          # Near-duplicate detection
│   ├── selector.py              # Selection and ranking
│   ├── album_curator.py         # Balanced ceremony distribution
│   ├── image_embedder.py        # CLIP-based image embeddings (NEW!)
│   ├── image_clusterer.py       # Smart clustering with K-Means/DBSCAN (NEW!)
│   ├── person_detector.py       # Person detection & categorization (NEW!)
│   └── verbose_logger.py        # Comprehensive logging & organization (NEW!)
├── test/
│   └── test_components.py       # Component tests
└── requirements.txt             # Enhanced with PyTorch, Transformers, scikit-learn
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

3. Find results in `album_selection_enhanced/`:
   - `final_selection/` - Selected images organized by person/role:
     - `01_bride/` - Best bride photos
     - `02_groom/` - Best groom photos
     - `03_couple/` - Best couple photos
     - `04_bride_parents/` - Bride's family
     - `05_bride_siblings/` - Bride's siblings
     - `06_groom_parents/` - Groom's family
     - `07_family/` - Extended family
     - `08_ceremony/` - Ceremony-focused shots
     - And more...
   - `intermediate_results/` - Saved data from each pipeline phase
   - `logs/` - Detailed logs and scoring reports
   - `SUMMARY_REPORT.txt` - Comprehensive summary

## Configuration

Edit `config.py` to customize:

### Basic Settings
- `SOURCE_FOLDER`: Path to your wedding photos (default: `./album_pics`)
- `OUTPUT_DIR`: Output directory (default: `./album_selection_enhanced`)
- `NUM_SELECT`: Total images to select (default: 1000)
- `BATCH_SIZE`: Processing batch size (default: 5)
- `LLM_PREFILTER_TARGET`: Images to send to LLM (default: 2000)
- `OPENAI_VISION_MODEL`: LLM model (default: gpt-4o)

### Enhanced Features
- `USE_EMBEDDINGS`: Enable CLIP embeddings (default: True)
- `USE_PERSON_DETECTION`: Enable person categorization (default: True)
- `USE_BALANCED_CURATION`: Ceremony-balanced selection (default: True)
- `EMBEDDING_MODEL`: CLIP model name (default: clip-ViT-B-32)
- `CLUSTERING_METHOD`: Clustering algorithm (kmeans/dbscan/auto)
- `NUM_CLUSTERS`: Number of clusters (auto if None)
- `IMAGES_PER_CLUSTER`: Best images per cluster (default: 10)
- `VERBOSE_LOGGING`: Save intermediate results (default: True)

### Person-Specific Quotas
Configure how many photos to select for each category in `PERSON_QUOTAS`:
- Bride: 150 photos
- Groom: 150 photos
- Couple: 200 photos
- Bride's/Groom's family: 50-30 each
- And more...

## How It Works (Enhanced 7-Phase Pipeline)

### Phase 1: Image Discovery & Prefiltering
- Scans source folder for all supported images (.jpg, .jpeg, .png)
- Calculates quality metrics (sharpness, exposure, colorfulness)
- Detects faces using Haar cascades
- Removes near-duplicates using perceptual hashing
- Keeps best image per hash
- **Saves**: `01_prefiltered_candidates.json`

### Phase 2: Image Embedding Generation
- Loads images and resizes for efficiency
- Generates CLIP embeddings using sentence-transformers
- Uses GPU/MPS if available for speed
- **Saves**: `02_embeddings.npz` (compressed numpy format)

### Phase 3: Smart Clustering
- Reduces dimensionality using UMAP (512 → 50 dimensions)
- Performs K-Means clustering with automatic cluster count optimization
- Uses silhouette score to find optimal number of clusters
- Selects best 10 representatives from each cluster
- Marks cluster representatives for bonus scoring
- **Saves**: `03_clustering_results.json` with cluster assignments

### Phase 4: LLM Quality Scoring
- Creates thumbnails for efficient processing
- Sends batches to GPT-4o vision model with specialized prompts
- Scores on:
  - **Technical Quality**: Sharpness, exposure, color accuracy
  - **Composition**: Framing, rule of thirds, creative angles
  - **Moment Value**: Identifies key ceremonies (Pheras, Varmala, Haldi, Mehendi, Sangeet, Vidaai)
  - **Cultural Details**: Recognizes jewelry, mehendi patterns, traditional attire
  - **Emotional Impact**: Detects genuine emotions, tears, joy, candid moments
- Tags images by ceremony type, subject, and moment
- **Saves**: `04_llm_scored_results.json`

### Phase 5: Person Detection & Categorization
- Uses LLM vision to identify people in photos:
  - Bride, Groom, Couple
  - Bride's mom, dad, parents, siblings
  - Groom's mom, dad, parents
  - Family groups, guests, children
- Assigns confidence scores to detections
- Categorizes images into person-specific groups
- **Saves**: `person_categorization.json`

### Phase 6: Final Selection & Ranking
- Combines scores:
  - LLM score (70%)
  - Local quality score (20%)
  - Cluster representative bonus (10%)
- Applies intelligent curation for ceremony balance
- Ensures diverse representation across:
  - Key Rituals (35%)
  - Emotional Moments (20%)
  - Pre-Wedding (15%)
  - Details (10%)
  - Portraits (12%)
  - Other (8%)

### Phase 7: Organization & Saving
- Organizes selected images into person-specific directories
- Applies quotas for each person category
- Names files with rank prefix (001_, 002_, etc.)
- Generates comprehensive CSV report with all scores
- Creates summary report with pipeline statistics
- **Saves**: Images organized by person + detailed reports

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

### AlbumCurator
Ensures balanced representation of different ceremony types in the final album. Prevents over-representation of any single ceremony and ensures all key moments are captured.

### Selector
Handles final ranking, selection, and output operations including CSV generation and file copying.

### ImageEmbedder (NEW!)
Generates semantic image embeddings using CLIP (Contrastive Language-Image Pre-training):
- Uses sentence-transformers for efficient embedding generation
- Supports GPU/MPS acceleration
- Batch processing for memory efficiency
- Saves/loads embeddings for reusability
- Computes similarity between images

### ImageClusterer (NEW!)
Groups similar images using advanced clustering algorithms:
- K-Means clustering with automatic optimal k determination
- DBSCAN for density-based clustering
- UMAP dimensionality reduction for better clustering
- Silhouette score optimization
- Selects representative images from each cluster
- Ensures diversity in final selection

### PersonDetector (NEW!)
AI-powered person identification and categorization:
- Uses LLM vision to identify people in photos
- Recognizes 14+ person categories:
  - Bride, Groom, Couple
  - Bride's/Groom's parents (individual and together)
  - Bride's siblings
  - Family groups, guests, children, ceremony focus
- Provides confidence scores for each detection
- Context-aware (uses clothing, jewelry, positioning)
- Batch processing for efficiency

### VerboseLogger (NEW!)
Comprehensive logging and organization system:
- Creates person-specific output directories
- Saves intermediate results at each pipeline phase
- Generates detailed execution logs
- Produces summary reports with statistics
- Tracks pipeline performance metrics
- Organizes images by detected person categories

## Customization

The modular design allows easy customization:

- Replace `ImageLoader` to load from cloud storage
- Modify `ImageProcessor` to add custom quality metrics
- Swap `LLMScorer` with different AI models
- Extend `Selector` with custom ranking algorithms

## License

MIT License - feel free to use and modify as needed.
