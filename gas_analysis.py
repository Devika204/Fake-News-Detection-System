import matplotlib.pyplot as plt

# -----------------------------
# READ REAL DATA
# -----------------------------
with open("gas_log.txt", "r") as f:
    gas_values = [int(line.strip()) for line in f.readlines()]

# Labels automatically
versions = [f"Run {i+1}" for i in range(len(gas_values))]

# -----------------------------
# PRINT TABLE
# -----------------------------
print("\n📊 REAL GAS DATA\n")

for v, g in zip(versions, gas_values):
    print(f"{v}: {g}")

# -----------------------------
# GRAPH
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(versions, gas_values, marker='o', color='green')

plt.title("Real Gas Usage Comparison")
plt.xlabel("Experiment Run")
plt.ylabel("Gas Used")

for i, v in enumerate(gas_values):
    plt.text(i, v + 1000, str(v), ha='center')

plt.grid(True)
plt.tight_layout()

plt.savefig("real_gas_comparison.png")
plt.show()