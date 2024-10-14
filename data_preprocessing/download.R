library(httr)
library(jsonlite)

base_url <- "https://api.ncbi.nlm.nih.gov/datasets/v1/genome/taxon/10029/download"

response <- GET(base_url, query = list(
  refseq = "true", 
  format = "fasta",  
  filename = "cricetulus_genome.zip"  # Specify output filename
))

if (status_code(response) == 200) {
  writeBin(content(response, "raw"), "cricetulus_genome.zip")
  cat("Download complete. File saved as 'cricetulus_genome.zip'.\n")
} else {
  cat("Error: Failed to download data. Status code:", status_code(response), "\n")
}
