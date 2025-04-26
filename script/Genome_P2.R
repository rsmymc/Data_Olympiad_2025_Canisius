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

# Step 1: Sample 10 buildings
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


#Trying more things to see missingness by grouping

# Step 1: Make building info table (site, type, tag)
building_info <- tibble(
  building = colnames(electricity_cleaned)[-1]  # exclude timestamp
) %>%
  separate(building, into = c("site", "type", "tag"), sep = "_", remove = FALSE)

# Step 2: Calculate % of missing values per building
missing_perc_building <- colMeans(is.na(electricity_cleaned[, -1]))
missing_perc_building <- tibble(building = names(missing_perc_building),
                                missing_perc = missing_perc_building)

# Step 3: Combine info
building_missing_summary <- building_info %>%
  left_join(missing_perc_building, by = "building")


#Grouping by site
missing_by_site <- building_missing_summary %>%
  group_by(site) %>%
  summarise(avg_missing_perc = mean(missing_perc, na.rm = TRUE)) %>%
  arrange(desc(avg_missing_perc))

#Grouping by type
missing_by_type <- building_missing_summary %>%
  group_by(type) %>%
  summarise(avg_missing_perc = mean(missing_perc, na.rm = TRUE)) %>%
  arrange(desc(avg_missing_perc))

# Missingness by Site
ggplot(missing_by_site, aes(x = reorder(site, -avg_missing_perc), y = avg_missing_perc)) +
  geom_col(fill = "steelblue") +
  labs(title = "Average Missingness by Site",
       x = "Site",
       y = "Average % Missing Data") +
  coord_flip()

# Missingness by Building Type
ggplot(missing_by_type, aes(x = reorder(type, -avg_missing_perc), y = avg_missing_perc)) +
  geom_col(fill = "darkorange") +
  labs(title = "Average Missingness by Building Type",
       x = "Building Type",
       y = "Average % Missing Data") +
  coord_flip()


#Swan column analysis
# Step 1: Select only columns that start with "Swan"
swan_columns <- electricity_cleaned %>%
  select(starts_with("Swan"))

# Step 2: Count the total NA values in these Swan columns
total_na_swan <- sum(is.na(swan_columns))

percent_na_swan <- colMeans(is.na(swan_columns)) * 100
#View(percent_na_swan)


#Plot
tibble(building = names(percent_na_swan), missing_perc = percent_na_swan) %>%
  ggplot(aes(x = reorder(building, -missing_perc), y = missing_perc)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(title = "Missing % for Swan Buildings",
       x = "Building",
       y = "% Missing")

# Buildings under the Swan site were found to have approximately 50% missing electricity readings.
# Given the high level of missingness, these buildings were excluded from the final dataset to ensure the integrity and reliability of the modeling process.
# Based on this insights we decide to drop it

# Remove all columns starting with "Swan"
electricity_cleaned <- electricity_cleaned %>%
  select(-starts_with("Swan"))

# Meter sensor noise cleanup
# Define a small threshold 
small_value_threshold <- 0.01  # Anything below 0.01 will be treated as near-zero

# Replace near-zero values with NA (excluding timestamp)
electricity_cleaned[, -1] <- electricity_cleaned[, -1] %>%
  mutate(across(everything(), ~ ifelse(!is.na(.x) & .x < small_value_threshold, NA, .x)))

# === Identify abnormal buildings based on average energy usage ===

# Calculate average and standard deviation of each building (column-wise)
building_means <- colMeans(electricity_cleaned[, -1], na.rm = TRUE)
building_sd <- apply(electricity_cleaned[, -1], 2, sd, na.rm = TRUE)

# Create summary table
building_summary <- tibble(
  building = names(building_means),
  mean_usage = building_means,
  sd_usage = building_sd
)

# === Identify which types of buildings consume the most energy ===

# Join building_summary with building_info (site, type, tag already separated earlier)
building_summary_full <- building_info %>%
  left_join(building_summary, by = "building")

# Group by building type and calculate average usage
type_summary <- building_summary_full %>%
  group_by(type) %>%
  summarise(avg_energy_usage = mean(mean_usage, na.rm = TRUE)) %>%
  arrange(desc(avg_energy_usage))

type_summary <- type_summary %>%
  filter(type != "unknown")

# Plot average energy usage by building type with gradient and highest on top
ggplot(type_summary, aes(x = reorder(type, avg_energy_usage), y = avg_energy_usage, fill = avg_energy_usage)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
  labs(
    title = "Average Energy Usage by Building Type",
    subtitle = "Darker Colors Represent Higher Energy Consumption",
    x = "Building Type",
    y = "Average Meter Reading (kWh)",
    fill = "Avg Usage"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 14, margin = margin(b = 10)),
    axis.title = element_text(face = "bold"),
    legend.position = "right",
    panel.grid.major.y = element_blank()
  )


# --- Plot Top 20 Highest Usage Buildings ---
top20_buildings <- building_summary %>%
  arrange(desc(mean_usage)) %>%
  slice_head(n = 20)

ggplot(top20_buildings, aes(x = reorder(building, mean_usage), y = mean_usage, fill = mean_usage)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +    # Gradient from low to high
  labs(
    title = "Top 20 Buildings by Abnormal Energy Usage",
    subtitle = "Higher Energy Users Highlighted with Darker Colors",
    x = "Building",
    y = "Average Meter Reading (kWh)",
    fill = "Average Usage"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 14, margin = margin(b = 10)),
    axis.title = element_text(face = "bold"),
    legend.position = "right",
    panel.grid.major.y = element_blank()  # Clean up horizontal gridlines
  )


# suggesting possible energy waste by timestamp
# Step 1: Create total energy usage per timestamp
electricity_time_avg <- electricity_cleaned %>%
  mutate(avg_usage = rowMeans(select(., -timestamp), na.rm = TRUE)) %>%
  select(timestamp, avg_usage)

# Extract hour of day and day of week
electricity_time_avg <- electricity_time_avg %>%
  mutate(hour = hour(timestamp),
         weekday = wday(timestamp, label = TRUE),  # Sunday = 1
         is_weekend = if_else(weekday %in% c("Sat", "Sun"), TRUE, FALSE))

usage_by_hour <- electricity_time_avg %>%
  group_by(hour) %>%
  summarise(avg_hourly_usage = mean(avg_usage, na.rm = TRUE))

ggplot(usage_by_hour, aes(x = hour, y = avg_hourly_usage)) +
  geom_line(color = "royalblue", size = 1.2) +               
  geom_point(color = "darkred", size = 2) +                  
  labs(title = "Average Energy Usage by Hour of Day",
       subtitle = "Hourly Energy Consumption Trends",
       x = "Hour of Day",
       y = "Average Meter Reading (kWh)") +
  theme_minimal(base_size = 14) +                           
  scale_x_continuous(breaks = 0:23) +                        
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 14, margin = margin(b = 10)),
    axis.title = element_text(face = "bold")
  )

# Average energy usage was calculated by hour of day and day type (weekday/weekend).
# Analysis revealed that certain buildings continue consuming high amounts of energy during off-hours (e.g., nighttime and weekends),
# suggesting possible energy waste. These insights can support operational changes to improve efficiency

# === Analyze impact of external factor: Temperature (Weather Data) ===

# Merge electricity_time_avg (timestamp, avg_usage) with weather (timestamp, temperature)
# Assuming weather has a "timestamp" and "temperature" column
electricity_weather <- electricity_time_avg %>%
  left_join(weather, by = "timestamp")

# Filter to remove missing airTemperature or avg_usage
electricity_weather_clean <- electricity_weather %>%
  filter(!is.na(airTemperature), !is.na(avg_usage))

# Plot: Energy Usage vs Air Temperature
ggplot(electricity_weather_clean, aes(x = airTemperature, y = avg_usage)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Relationship between Energy Usage and Air Temperature",
       x = "Air Temperature (°C or °F)",
       y = "Average Meter Reading (kWh)")


# Recalculate electricity_final after all cleaning is done
threshold <- ncol(electricity_cleaned) * 0.9

electricity_final <- electricity_cleaned %>%
  filter(rowSums(is.na(.)) < threshold)

# Save the cleaned dataset to CSV
write_csv(electricity_final, "data/electricity_final_clean_V2.csv")

