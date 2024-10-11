library(httr)
library(jsonlite)

# Define the base URL for the NCBI datasets API
base_url <- "https://api.ncbi.nlm.nih.gov/datasets/v1/genome/taxon/10029/download"

# Make an API request to download the genomic data
response <- GET(base_url, query = list(
  refseq = "true",  # Reference genomes only
  format = "fasta",  # Download in FASTA format
  filename = "cricetulus_genome.zip"  # Specify output filename
))

# Check if the request was successful
if (status_code(response) == 200) {
  # Write the content to a ZIP file
  writeBin(content(response, "raw"), "cricetulus_genome.zip")
  cat("Download complete. File saved as 'cricetulus_genome.zip'.\n")
} else {
  cat("Error: Failed to download data. Status code:", status_code(response), "\n")
}
