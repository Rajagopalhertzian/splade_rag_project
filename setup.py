from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

requirements = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "accelerate>=0.24.0",
    "sentence-transformers>=2.2.0",
    "huggingface-hub>=0.19.0",
    "safetensors>=0.4.0",
    "elasticsearch>=8.10.0",
    "rank-bm25>=0.2.2",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
    "numpy>=1.24.0",
]

extras_require = {
    "train": ["datasets>=2.14.0", "evaluate>=0.4.0", "scikit-learn>=1.3.0"],
    "web": ["streamlit>=1.28.0", "fastapi>=0.104.0", "uvicorn>=0.24.0"],
    "dev": ["pytest>=7.4.0", "black>=23.0.0", "ruff>=0.1.0"],
    "all": ["datasets>=2.14.0", "evaluate>=0.4.0", "scikit-learn>=1.3.0",
            "streamlit>=1.28.0", "fastapi>=0.104.0", "uvicorn>=0.24.0",
            "pytest>=7.4.0", "black>=23.0.0", "ruff>=0.1.0"],
}

setup(
    name="splade-rag",
    version="1.0.0",
    author="RAG Project",
    author_email="rag@example.com",
    description="SPLADE-based Retrieval-Augmented Generation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/splade-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "splade-build-index=scripts.build_index:main",
            "splade-query=scripts.query_rag:main",
            "splade-train=scripts.train_splade:main",
            "splade-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "splade_rag": ["../config/*.yaml", "../docs/*.doc"],
    },
)