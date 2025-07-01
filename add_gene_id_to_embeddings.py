#!/usr/bin/env python3
"""
Script to add EnsemblGeneID column to all_embeddings.csv and convert to TSV format.
"""

import pandas as pd
import sys
from pathlib import Path


def main():
    # File paths
    table_ev6_path = "data/paper/Table_EV6.tsv"
    embeddings_csv_path = "data/rinalmo/all_embeddings.csv"
    output_path = "data/rinalmo/all_embeddings_with_gene_id.tsv"

    print("🔍 Loading Table_EV6 for transcript to gene ID mapping...")

    # Read Table_EV6 to get the transcript ID -> gene ID mapping
    # We only need the EnsemblGeneID and EnsemblTranscriptID columns
    try:
        table_ev6 = pd.read_csv(
            table_ev6_path, sep="\t", usecols=["EnsemblGeneID", "EnsemblTranscriptID"]
        )
        print(f"   ✅ Loaded {len(table_ev6)} transcript-gene mappings")
    except Exception as e:
        print(f"   ❌ Error loading Table_EV6: {e}")
        return 1

    # Create mapping dictionary for faster lookup
    transcript_to_gene = dict(
        zip(table_ev6["EnsemblTranscriptID"], table_ev6["EnsemblGeneID"])
    )
    print(f"   📚 Created mapping dictionary with {len(transcript_to_gene)} entries")

    print("\n📊 Loading embeddings file...")

    # Read the embeddings CSV file
    try:
        embeddings_df = pd.read_csv(embeddings_csv_path)
        print(f"   ✅ Loaded embeddings with shape: {embeddings_df.shape}")
        print(
            f"   📋 Columns: {list(embeddings_df.columns[:5])}..."
        )  # Show first 5 column names
    except Exception as e:
        print(f"   ❌ Error loading embeddings: {e}")
        return 1

    print("\n🔗 Adding EnsemblGeneID column...")

    # Add EnsemblGeneID column by mapping EnsemblTranscriptID
    embeddings_df["EnsemblGeneID"] = embeddings_df["EnsemblTranscriptID"].map(
        transcript_to_gene
    )

    # Check for any missing mappings
    missing_mappings = embeddings_df["EnsemblGeneID"].isna().sum()
    if missing_mappings > 0:
        print(
            f"   ⚠️  Warning: {missing_mappings} transcript IDs could not be mapped to gene IDs"
        )
        # Show some examples of unmapped transcripts
        unmapped = embeddings_df[embeddings_df["EnsemblGeneID"].isna()][
            "EnsemblTranscriptID"
        ].head(5)
        print(f"   📝 Examples of unmapped transcripts: {list(unmapped)}")
    else:
        print(f"   ✅ All {len(embeddings_df)} transcript IDs successfully mapped!")

    # Reorder columns to put EnsemblGeneID right after GeneName and EnsemblTranscriptID
    columns = list(embeddings_df.columns)
    # Find the index of EnsemblTranscriptID
    transcript_idx = columns.index("EnsemblTranscriptID")
    # Remove EnsemblGeneID from its current position (last)
    columns.remove("EnsemblGeneID")
    # Insert it right after EnsemblTranscriptID
    columns.insert(transcript_idx + 1, "EnsemblGeneID")

    # Reorder the dataframe
    embeddings_df = embeddings_df[columns]

    print(f"\n💾 Saving to TSV format: {output_path}")

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as TSV
    try:
        embeddings_df.to_csv(output_path, sep="\t", index=False)
        print(f"   ✅ Successfully saved {len(embeddings_df)} rows to {output_path}")
        print(f"   📏 Final shape: {embeddings_df.shape}")
        print(f"   📋 First few columns: {list(embeddings_df.columns[:5])}")
    except Exception as e:
        print(f"   ❌ Error saving file: {e}")
        return 1

    # Print summary statistics
    print(f"\n📈 Summary:")
    print(f"   • Total transcripts processed: {len(embeddings_df)}")
    print(f"   • Successfully mapped: {len(embeddings_df) - missing_mappings}")
    print(f"   • Missing mappings: {missing_mappings}")
    print(f"   • Total columns: {len(embeddings_df.columns)}")
    print(f"   • Output file: {output_path}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
