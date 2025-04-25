rm(list = ls())

library(tidyverse)
library(lubridate)
library(naniar)
library(dplyr)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load datasets from the "data" folder
electricity <- read_csv("data/electricity.csv")
metadata <- read_csv("data/metadata.csv")
weather <- read_csv("data/weather.csv")

# Dimensions of the dataset
dim(electricity)

# Preview column names only
colnames(electricity)[1:10]  # First 10 column names

# Count NAs in just a few columns
colSums(is.na(electricity[, 1:10]))

# # Convert timestamp to proper datetime
# electricity <- electricity %>%
#   mutate(timestamp = ymd_hms(timestamp)) #Caused by warning: !  731 failed to parse. 

head(electricity$timestamp, 20)

#Check to see bad timestamps
bad_timestamps <- electricity$timestamp[is.na(ymd_hms(electricity$timestamp))]
length(bad_timestamps)  # 731
unique(bad_timestamps)  # NA

#Check to see if its okay to drop or impute
electricity_na_ts <- electricity %>%
  filter(is.na(timestamp))

#View(electricity_na_ts)

# After analysing the data, it has been found that timestamps are missing at the 24th hour. Would be imputing it.

# Step 0: Ensure timestamp is parsed (even if NA)
electricity <- electricity %>%
  mutate(timestamp = ymd_hms(timestamp))

# Step 1: Identify rows with NA timestamps
na_time_rows <- which(is.na(electricity$timestamp))

# Step 2: For each NA, assign the timestamp as 1 hour after the previous row
# We'll make a copy to modify
electricity_fixed <- electricity

for (i in na_time_rows) {
  if (i > 1 && !is.na(electricity_fixed$timestamp[i - 1])) {
    electricity_fixed$timestamp[i] <- electricity_fixed$timestamp[i - 1] + hours(1)
  } else if (i < nrow(electricity_fixed) && !is.na(electricity_fixed$timestamp[i + 1])) {
    electricity_fixed$timestamp[i] <- electricity_fixed$timestamp[i + 1] - hours(1)
  } else {
    # If both neighbors are NA or missing, leave it NA
    electricity_fixed$timestamp[i] <- NA
  }
}

sum(is.na(electricity_fixed$timestamp))

# Since the meter reading cannot be absolute zero, could be that meter wasn't working at that time.
# We will replace that with NA

# Replace all 0s with NA
electricity_cleaned <- electricity_fixed

electricity_cleaned[, -1] <- electricity_cleaned[, -1] %>%
  mutate(across(everything(), ~ ifelse(.x == 0, NA, .x)))

# Sampling a few buildings 

few_buildings <- sample(colnames(electricity_cleaned)[-1], 30)
electricity_small <- electricity_cleaned %>%
  select(timestamp, all_of(few_buildings)) %>%
  slice_sample(n = 2500)

vis_miss(electricity_small, cluster = TRUE)

# # To visualize missing data patterns without overwhelming memory, a random sample was extracted from the full dataset.
# Specifically, 25 building sites were randomly selected, and 2,500 timestamps were randomly sampled from the electricity data.
# Based on this subset, approximately 10% of the entries were missing (NA) and 90% of the entries were present.

set.seed(123)

# Step 1: Sample 25 buildings
sampled_buildings <- sample(colnames(electricity_cleaned)[-1], 10)

electricity_small <- electricity_cleaned %>%
  select(timestamp, all_of(sampled_buildings))

# Step 2: Sample 2500 timestamps
electricity_small <- electricity_small %>%
  slice_sample(n = 2500)

# Histogram of all sampled building meter readings

electricity_small %>%
  pivot_longer(cols = -timestamp, names_to = "building", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 100, fill = "skyblue", color = "black") +
  scale_x_log10() +  # Log scale because readings are skewed
  labs(title = "Distribution of Meter Readings (Sampled Buildings)",
       x = "Meter Reading (log10 scale)",
       y = "Count")



# Line plots for sampled buildings over sampled timestamps
electricity_small %>%
  pivot_longer(cols = -timestamp, names_to = "building", values_to = "value") %>%
  ggplot(aes(x = timestamp, y = value, color = building)) +
  geom_line(alpha = 0.7) +
  labs(title = "Meter Readings Over Time (Sampled Buildings)",
       x = "Timestamp",
       y = "Meter Reading")

# Boxplot per building
electricity_small %>%
  pivot_longer(cols = -timestamp, names_to = "building", values_to = "value") %>%
  ggplot(aes(x = building, y = value)) +
  geom_boxplot(outlier.color = "red", outlier.size = 1) +
  coord_flip() +
  labs(title = "Boxplot of Meter Readings by Building",
       x = "Building",
       y = "Meter Reading")


# Distribution and boxplot analyses revealed that most buildings have consistent electricity usage, but a few show extreme outliers and irregular patterns.
# Additionally, near-zero values and missing data likely reflect sensor issues rather than real usage.
# Based on these insights, we will drop the rows in which it has 90% missing values.

#Setting threshold
threshold <- ncol(electricity_cleaned) * 0.9

electricity_final <- electricity_cleaned %>%
  filter(rowSums(is.na(.)) < threshold)

# Save the cleaned dataset to CSV
write_csv(electricity_final, "data/electricity_final_clean.csv")
