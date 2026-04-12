import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. Load the GKE Data
gke_csv = """Timestamp_Min,Concurrent_Users,Requests_Per_Sec,Failures_Per_Sec,P95_External_Latency_ms,Internal_GPU_Latency_ms,CPU_Usage_Percent,GPU_Usage_Percent,Memory_Usage_Percent
1,10,10,0,42,18,25,10,60
2,10,10,0,41,19,22,12,60
3,10,10,0,43,20,24,11,61
4,10,10,0,42,19,23,10,61
5,10,10,0,41,18,25,12,60
6,50,50,0,45,19,35,30,62
7,50,50,0,44,20,38,32,62
8,50,50,0,46,21,36,29,63
9,50,50,0,44,19,34,31,63
10,50,50,0,45,20,35,30,62
11,100,100,0,48,20,55,55,65
12,100,100,0,47,21,58,58,65
13,100,100,0,49,20,56,54,66
14,100,100,0,48,22,57,56,66
15,100,100,0,47,19,55,55,65
16,1000,800,45,5200,22,85,98,80
17,1000,800,50,5500,21,88,98,82
18,1000,800,48,5400,23,86,98,81
19,1000,950,5,210,20,78,88,78
20,1000,1000,0,52,21,75,85,75
"""

# 2. Load the Cloud Run Data
cloudrun_csv = """Timestamp_Min,Concurrent_Users,Requests_Per_Sec,Failures_Per_Sec,P95_External_Latency_ms,Internal_GPU_Latency_ms,CPU_Usage_Percent,GPU_Usage_Percent,Memory_Usage_Percent
1,10,8,0,15000,20,100,15,50
2,10,10,0,62,19,25,12,60
3,10,10,0,65,21,28,10,61
4,10,10,0,61,18,22,11,60
5,10,10,0,64,22,24,12,61
6,50,45,0,18000,21,100,35,65
7,50,50,0,68,19,45,30,66
8,50,50,0,71,23,42,28,66
9,50,50,0,66,18,48,32,65
10,50,50,0,69,20,44,30,66
11,100,90,0,22000,19,100,60,70
12,100,100,0,72,21,65,52,72
13,100,100,0,76,24,62,55,71
14,100,100,0,69,18,68,54,72
15,100,100,0,71,22,64,52,71
16,1000,900,15,28000,25,100,95,85
17,1000,1000,0,85,21,85,82,82
18,1000,1000,0,78,23,88,80,81
19,1000,1000,0,82,19,82,84,82
20,1000,1000,0,79,22,86,81,81
"""

df_gke = pd.read_csv(io.StringIO(gke_csv))
df_gke['Platform'] = 'GKE (Standard/Autopilot)'

df_cr = pd.read_csv(io.StringIO(cloudrun_csv))
df_cr['Platform'] = 'Cloud Run (Serverless)'

# Combine for the comparative plots
df_combined = pd.concat([df_gke, df_cr])

# Set visual style
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Performance & Resource Scaling: GKE vs. Cloud Run", fontsize=16, fontweight='bold')

# Helper function to format the x-axis uniformly
def format_xaxis(ax):
    # Set x-ticks explicitly to match our user load tiers
    ax.set_xticks([10, 50, 100, 1000])
    # Use log scale on x-axis because 100 to 1000 is a massive visual jump
    ax.set_xscale('log')
    ax.set_xticklabels(['10 Users', '50 Users', '100 Users', '1000 Users'])
    ax.set_xlabel('Concurrent User Requests')

# --- Plot 1: Throughput (RPS) ---
ax1 = plt.subplot(3, 2, 1)
sns.lineplot(data=df_combined, x='Concurrent_Users', y='Requests_Per_Sec', hue='Platform', marker='o', ax=ax1)
ax1.set_title('Throughput (Requests / Sec)', fontweight='bold')
ax1.set_ylabel('RPS')
format_xaxis(ax1)

# --- Plot 2: Failures ---
ax2 = plt.subplot(3, 2, 2)
sns.lineplot(data=df_combined, x='Concurrent_Users', y='Failures_Per_Sec', hue='Platform', marker='o', ax=ax2)
ax2.set_title('Dropped Requests (Failures / Sec)', fontweight='bold')
ax2.set_ylabel('Failures')
format_xaxis(ax2)

# --- Plot 3: External Latency ---
ax3 = plt.subplot(3, 2, 3)
sns.lineplot(data=df_combined, x='Concurrent_Users', y='P95_External_Latency_ms', hue='Platform', marker='o', ax=ax3)
ax3.set_title('P95 External Latency (Round-Trip)', fontweight='bold')
ax3.set_ylabel('Latency (ms)')
format_xaxis(ax3)

# --- Plot 4: Internal GPU Latency ---
ax4 = plt.subplot(3, 2, 4)
sns.lineplot(data=df_combined, x='Concurrent_Users', y='Internal_GPU_Latency_ms', hue='Platform', marker='o', ax=ax4)
ax4.set_title('Internal GPU Latency (Triton Model)', fontweight='bold')
ax4.set_ylabel('Latency (ms)')
format_xaxis(ax4)

# --- Prep data for Resources (Melting DataFrames for Seaborn) ---
gke_res = df_gke.melt(id_vars=['Concurrent_Users'], value_vars=['CPU_Usage_Percent', 'GPU_Usage_Percent', 'Memory_Usage_Percent'], var_name='Resource', value_name='Percent')
cr_res = df_cr.melt(id_vars=['Concurrent_Users'], value_vars=['CPU_Usage_Percent', 'GPU_Usage_Percent', 'Memory_Usage_Percent'], var_name='Resource', value_name='Percent')

# --- Plot 5: GKE Resources ---
ax5 = plt.subplot(3, 2, 5)
sns.lineplot(data=gke_res, x='Concurrent_Users', y='Percent', hue='Resource', marker='o', ax=ax5)
ax5.set_title('GKE Resource Usage', fontweight='bold')
ax5.set_ylabel('Usage (%)')
ax5.set_ylim(0, 105)
format_xaxis(ax5)

# --- Plot 6: Cloud Run Resources ---
ax6 = plt.subplot(3, 2, 6)
sns.lineplot(data=cr_res, x='Concurrent_Users', y='Percent', hue='Resource', marker='o', ax=ax6)
ax6.set_title('Cloud Run Resource Usage', fontweight='bold')
ax6.set_ylabel('Usage (%)')
ax6.set_ylim(0, 105)
format_xaxis(ax6)

plt.tight_layout()
plt.subplots_adjust(top=0.92) # Adjust to fit the main title
plt.show()