# Apache Spark Development Environment Setup Guide

## System Information
- **OS**: Windows
- **Python Version**: 3.11.9 ✓ (Compatible)
- **Target Spark Version**: 3.5.0
- **Target Java Version**: 11
- **Target Scala Version**: 2.12

---

## Step 1: Install Java 11 (IMPORTANT - Required for Spark)

### Option 1: Manual Download (Recommended)
1. Go to: https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html
2. Download **JDK 11.0.20** (or latest 11.x) for Windows x64
3. Install to: `C:\Java\jdk-11`
4. Accept the default installation settings

### Option 2: Using Chocolatey (If installed)
```powershell
choco install openjdk11 -y
```

### Option 3: Using Windows Package Manager
```powershell
winget install Oracle.OpenJDK.11
```

---

## Step 2: Set Up Java Environment Variables

After Java installation, configure system environment variables:

1. **Open Environment Variables**:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to "Advanced" tab → "Environment Variables" button

2. **Create JAVA_HOME Variable**:
   - Click "New" under System variables
   - Variable name: `JAVA_HOME`
   - Variable value: `C:\Java\jdk-11` (or your installation path)
   - Click OK

3. **Update PATH Variable**:
   - Select "Path" → Click "Edit"
   - Click "New" and add: `%JAVA_HOME%\bin`
   - Click OK on all dialogs

4. **Verify Installation** (Open new PowerShell/CMD):
   ```powershell
   java -version
   javac -version
   ```
   Both should show version 11

---

## Step 3: Download and Install Apache Spark 3.5.0

### Manual Installation (Recommended)

1. **Download Spark**:
   - Go to: https://spark.apache.org/downloads.html
   - Select: Package type = `Pre-built for Apache Hadoop 3.3 and later`
   - Click to download the `.tgz` file

2. **Extract Spark**:
   - Download location: anywhere (we'll use `C:\spark`)
   - Install 7-Zip or WinRAR if needed
   - Extract the `.tgz` file to `C:\spark\spark-3.5.0`

3. **Verify Extraction**:
   ```
   C:\spark\spark-3.5.0\
   ├── bin\
   ├── conf\
   ├── data\
   ├── jars\
   └── python\
   ```

---

## Step 4: Configure PySpark Environment Variables

1. **Open Environment Variables** (as before):
   - Press `Win + R`, type `sysdm.cpl`, press Enter

2. **Create SPARK_HOME Variable**:
   - Variable name: `SPARK_HOME`
   - Variable value: `C:\spark\spark-3.5.0`

3. **Create PYSPARK_PYTHON Variable**:
   - Variable name: `PYSPARK_PYTHON`
   - Variable value: `python` (or full path: `C:\Users\HP 1040 G7\AppData\Local\Programs\Python\Python311\python.exe`)

4. **Update PATH** (if not already done):
   - Add: `%SPARK_HOME%\bin` to PATH

5. **Verify** (Open new PowerShell):
   ```powershell
   echo $env:SPARK_HOME
   echo $env:JAVA_HOME
   spark-shell --version
   ```

---

## Step 5: Install Required Python Packages

Open PowerShell and run:

```powershell
pip install --upgrade pip setuptools wheel
pip install pyspark==3.5.0
pip install jupyter notebook jupyterlab
pip install pandas numpy matplotlib seaborn
pip install scikit-learn xgboost lightgbm
```

### Verify PySpark Installation:
```powershell
python -c "import pyspark; print(pyspark.__version__)"
```

---

## Step 6: Configure Spark for Local Development

Create a configuration file at `C:\spark\spark-3.5.0\conf\spark-defaults.conf`:

```properties
# Spark Configuration for Local Development
spark.driver.memory              4g
spark.driver.maxResultSize       2g
spark.executor.memory            4g
spark.network.timeout            600s
spark.sql.shuffle.partitions     200
```

---

## Step 7: Test Spark Installation

Create a test script at: `test_spark.py`

```python
from pyspark.sql import SparkSession

# Create Spark Session
spark = SparkSession.builder \
    .appName("PySpark Test") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Get Spark Context
sc = spark.sparkContext

# Test 1: Simple RDD operation
data = range(1, 101)
rdd = sc.parallelize(data)
result = rdd.map(lambda x: x * 2).sum()
print(f"✓ Test 1 Passed: Sum of doubled values (1-100) = {result}")

# Test 2: DataFrame operation
df = spark.createDataFrame([(1, "Alice"), (2, "Bob"), (3, "Charlie")], ["id", "name"])
print(f"✓ Test 2 Passed: Created DataFrame with {df.count()} rows")
df.show()

# Test 3: Spark SQL
df.createOrReplaceTempView("people")
result_df = spark.sql("SELECT * FROM people WHERE id > 1")
print(f"✓ Test 3 Passed: SQL query returned {result_df.count()} rows")

print("\n✅ All tests passed! Spark is ready to use.")
spark.stop()
```

Run the test:
```powershell
python test_spark.py
```

---

## Step 8: Create Jupyter Notebook Kernel for Spark

1. **Create Kernel Spec Directory**:
   ```powershell
   $kernelPath = "$env:APPDATA\jupyter\kernels\pyspark"
   mkdir $kernelPath -Force
   ```

2. **Create kernel.json** in that directory:

   ```json
   {
     "display_name": "PySpark 3.5.0",
     "language": "python",
     "argv": [
       "python",
       "-m",
       "ipykernel_launcher",
       "-f",
       "{connection_file}"
     ],
     "env": {
       "SPARK_LOCAL_IP": "127.0.0.1",
       "PYSPARK_PYTHON": "python"
     }
   }
   ```

3. **Start Jupyter**:
   ```powershell
   jupyter notebook
   ```

---

## Troubleshooting

### Issue: "Java not found"
- Verify JAVA_HOME is set: `echo %JAVA_HOME%`
- Restart PowerShell/CMD after setting environment variables

### Issue: "spark-shell command not found"
- Verify SPARK_HOME is set and bin is in PATH
- Restart PowerShell after setting environment variables

### Issue: PySpark not importing
- Verify: `python -c "import pyspark; print(pyspark.__version__)"`
- If fails, reinstall: `pip install --force-reinstall pyspark==3.5.0`

### Issue: Out of Memory errors
- Increase memory in `spark-defaults.conf`
- Reduce parallelism: `spark.sql.shuffle.partitions = 100`

---

## Next Steps

After successful setup:
1. Create a Spark test notebook
2. Explore Spark DataFrames and SQL
3. Begin data ingestion for retail analytics
4. Implement feature engineering pipeline
5. Build predictive models with MLlib

