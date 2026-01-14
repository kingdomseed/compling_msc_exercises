import pandas as pd
import numpy as np
import os

# Define columns
columns = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'
]

# Hardcoded data for rows 0-4
head_data = [
    [0.587349, 0.577747, 0.386298, 0.568199, 0.647347, 0.292985, 0.654522, 0.795029, 0.630115, 0],
    [0.643654, 0.441300, 0.314381, 0.439304, 0.514545, 0.356685, 0.377248, 0.202914, 0.520358, 0],
    [0.388934, 0.470876, 0.506122, 0.524364, 0.561537, 0.142913, 0.249922, 0.401487, 0.219973, 0],
    [0.725820, 0.715942, 0.506141, 0.521683, 0.751819, 0.148683, 0.467200, 0.658678, 0.242428, 0],
    [0.610517, 0.532588, 0.237701, 0.270288, 0.495155, 0.494792, 0.409721, 0.469762, 0.585049, 0]
]

# Hardcoded data for rows 2006-2010
tail_data = [
    [0.636224, 0.580511, 0.277748, 0.418063, 0.522486, 0.342184, 0.310364, 0.402799, 0.627156, 1],
    [0.470143, 0.548826, 0.301347, 0.538273, 0.498565, 0.231359, 0.565061, 0.175889, 0.395061, 1],
    [0.817826, 0.087434, 0.656389, 0.670774, 0.369089, 0.431872, 0.563265, 0.285745, 0.578674, 1],
    [0.424187, 0.464092, 0.459656, 0.541633, 0.615572, 0.388360, 0.397780, 0.449156, 0.440004, 1],
    [0.322425, 0.492891, 0.841409, 0.492136, 0.656047, 0.588709, 0.471422, 0.503458, 0.591867, 1]
]

# Total rows desired: indices 0 to 2010 -> 2011 rows
total_rows = 2011
rows_generated = 0

all_data = []

# Add head data
all_data.extend(head_data)
rows_generated += len(head_data)

# Generate middle data
# Rows 5 to 2005 (inclusive) -> 2006 - 5 = 2001 rows needed
middle_count = 2006 - 5
print(f"Generating {middle_count} random rows...")

# Random values between 0 and 1 for features
# For Potability, we can randomize 0 or 1, or just set 0 for half and 1 for half?
# The image shows Potability is 0 at start and 1 at end.
# Let's assume a split or just random. Random integer 0 or 1 seems appropriate.

np.random.seed(42) # For reproducibility
random_features = np.random.rand(middle_count, 9) # 9 feature columns
random_potability = np.random.randint(0, 2, size=(middle_count, 1))

middle_data = np.hstack((random_features, random_potability))

all_data.extend(middle_data.tolist())
rows_generated += len(middle_data)

# Add tail data
all_data.extend(tail_data)
rows_generated += len(tail_data)

print(f"Total rows generated: {rows_generated}")

# Create DataFrame
df = pd.DataFrame(all_data, columns=columns)

# Verify types (Potability should be int, others float)
df['Potability'] = df['Potability'].astype(int)

# Save to CSV
output_file = 'water_potability.csv'
df.to_csv(output_file, index=False) # index=False because the image index seems to be just row numbers, or maybe part of the data?
# The image shows an index column '0', '1', etc. But usually CSVs don't save the index unless requested.
# The user asked for a "csv file for a model", usually checking for pandas default read_csv adds an index.
# The image has a bold index column on the left.
# If I save index=True, it will add an unnamed index column.
# Let's save index=True but with a specific name if needed? 
# Usually 'index=False' is safer for ML datasets unless the ID is meaningful.
# Wait, look at the image again. 
# "0 0.587..." 
# The first column is bold indices 0, 1, 2... 2010.
# I will output with index=True just in case, or let the user decide.
# Actually, the user just said "turn this fake dataset into a csv file".
# Most standard CSVs for ML don't include the row number as a feature.
# I will stick to index=False to keep it clean, the row number is implicit.

print(f"File saved to {os.path.abspath(output_file)}")

# Verification print
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
