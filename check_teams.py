import pandas as pd

# Load dataset
df = pd.read_csv("datasets/CRICKET.csv")

# Get all unique teams
teams = set(df['batting_team']).union(set(df['bowling_team']))

print("\n✅ UNIQUE TEAMS IN DATASET:\n")

for t in sorted(teams):
    print(t)

print("\n🔢 TOTAL TEAMS:", len(teams))


# OPTIONAL: find similar names (mistakes)
print("\n⚠️ CHECKING FOR POSSIBLE DUPLICATES:\n")

for t in sorted(teams):
    for t2 in sorted(teams):
        if t != t2 and t.lower().replace(" ", "") == t2.lower().replace(" ", ""):
            print("Possible duplicate:", t, "↔", t2)