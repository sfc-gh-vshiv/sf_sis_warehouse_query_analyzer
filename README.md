# ❄️ Snowflake Warehouse Usage Monitor & Query Analyzer

A comprehensive **Streamlit in Snowflake (SiS)** application for monitoring warehouse credit usage, detecting consumption spikes, and analyzing query performance with advanced predictive analytics and machine learning capabilities.

## 🚀 Streamlit in Snowflake (SiS) Application

This application runs natively within your Snowflake environment using **Streamlit in Snowflake**, providing:
- **Zero Infrastructure Management**: No external hosting required
- **Secure Data Access**: Direct access to ACCOUNT_USAGE views
- **Automatic Dependency Management**: Uses `environment.yaml` for seamless package installation
- **Native Snowflake Integration**: Leverages Snowpark for optimal performance

## 🌟 Key Features

### 📊 Executive Summary Dashboard
- **Real-time KPIs**: Current week metrics with week-over-week comparisons
- **Smart Insights Cards**: Most volatile warehouses, fastest growing, efficiency leaders
- **Custom Styled Metrics**: Professional cards with color-coded indicators
- **Quick Analytics**: Top performers and anomaly status at a glance

### 🚨 Advanced Spike Detection
- **Configurable Thresholds**: Adjustable spike detection (10-200% increase)
- **Top 6 Largest Spikes**: Visual cards with detailed spike information
- **Interactive Heatmap**: Week-over-week changes across all warehouses
- **Smart Filtering**: Option to show only warehouses with spikes
- **Zero Handling**: Displays hyphens for 0% changes with white backgrounds

### 📈 Trends & Forecasting
- **Prophet-Based Predictions**: Advanced time series forecasting (1-8 weeks ahead)
- **Interactive Charts**: Warehouse selection with confidence intervals
- **Trend Analysis**: Rolling averages and seasonal pattern detection
- **Forecasting Controls**: Configurable prediction periods and parameters

### 🔍 Deep Analytics
- **Machine Learning Anomaly Detection**: Isolation Forest algorithm with sensitivity controls
- **Statistical Analysis**: Comprehensive warehouse performance metrics
- **Usage Pattern Analysis**: Weekly and seasonal trend identification
- **Efficiency Rankings**: Performance comparisons with visual indicators
- **Dynamic Visualizations**: Scatter plots with anomaly severity indicators

### 🔍 Query-Level Analysis (NEW!)
- **Query Performance Metrics**: Execution times, data scanned, success rates
- **Warehouse Query Analysis**: Performance by warehouse with efficiency metrics
- **Query Type Distribution**: Visual breakdown of query types and patterns
- **User Activity Analysis**: Top users and role-based usage patterns
- **Efficiency Analysis**: Spillage rates, partition pruning effectiveness
- **Resource Utilization**: Credits per query, bytes scanned analysis

### 📋 Raw Data & Export
- **Advanced Filtering**: Multi-dimensional filtering with warehouse and credit thresholds
- **Interactive Tables**: Sortable, formatted data display
- **Export Capabilities**: CSV downloads with timestamp naming
- **Summary Statistics**: Real-time data aggregations

## 🛠️ Installation & Deployment

### Streamlit in Snowflake Setup

1. **Create the SiS Application**:
```sql
CREATE STREAMLIT app_warehouse_monitor
  ROOT_LOCATION = '@your_stage/warehouse_monitor'
  MAIN_FILE = 'streamlit_with_query_analysis.py'
  QUERY_WAREHOUSE = 'your_warehouse';
```

2. **Upload Application Files**:
```sql
PUT 'file://streamlit_with_query_analysis.py' @your_stage/warehouse_monitor AUTO_COMPRESS=FALSE;
PUT 'file://environment.yaml' @your_stage/warehouse_monitor AUTO_COMPRESS=FALSE;
```

3. **Dependencies (Automatic)**:
The `environment.yaml` file automatically installs all required packages:
```yaml
name: app_environment
channels:
  - snowflake
dependencies:
  - plotly=6.0.1
  - prophet=1.1.5
  - python=3.11.*
  - scikit-learn=1.6.1
  - scipy=1.15.3
  - snowflake-snowpark-python=
  - statsmodels=0.14.4
  - streamlit=
```

### Required Permissions

Grant the following privileges to your role:
```sql
-- Warehouse usage data
GRANT USAGE ON DATABASE SNOWFLAKE TO ROLE your_role;
GRANT USAGE ON SCHEMA SNOWFLAKE.ACCOUNT_USAGE TO ROLE your_role;
GRANT SELECT ON VIEW SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY TO ROLE your_role;

-- Query analysis data (for Query Analysis tab)
GRANT SELECT ON VIEW SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY TO ROLE your_role;
```

## 📊 Data Sources & Latency

### Primary Data Sources
- **`SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY`**
  - Hourly warehouse credit consumption
  - Up to 3 hours latency
  - Includes compute and cloud services credits

- **`SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY`**
  - Individual query performance metrics
  - Up to 45 minutes latency
  - Covers last 365 days of query activity

### Caching Strategy
- **Warehouse Data**: 10-minute cache for performance optimization
- **Query Data**: 5-minute cache for more recent insights
- **Smart Refresh**: Manual refresh available in sidebar

## 🎯 Feature Guide

### 1. Executive Summary
Start here for high-level insights:
- Monitor key metrics with week-over-week changes
- Review custom insight cards for quick wins
- Identify top performing and problematic warehouses

### 2. Spike Detection
Proactive monitoring capabilities:
- Adjust spike thresholds based on your tolerance
- Review visual spike cards for detailed analysis
- Use the heatmap to spot patterns across warehouses

### 3. Trends & Forecasting
Strategic planning tools:
- Select warehouses for detailed trend analysis
- Generate forecasts for capacity planning
- Use predictions for budget estimation

### 4. Deep Analytics
Advanced analysis and optimization:
- Review ML-powered anomaly detection
- Analyze efficiency metrics for optimization opportunities
- Compare warehouse performance rankings

### 5. Query Analysis
Granular performance insights:
- Monitor query-level performance metrics
- Identify inefficient queries and users
- Analyze spillage and partition pruning effectiveness

### 6. Raw Data
Data exploration and export:
- Apply custom filters for focused analysis
- Export filtered datasets for external analysis
- View comprehensive summary statistics

## ⚙️ Configuration Options

### Sidebar Controls
- **📅 Time Range**: 2-24 weeks of historical data
- **🔮 Forecasting**: 1-8 weeks ahead predictions
- **🚨 Spike Detection**: 10-200% threshold adjustment
- **🤖 Anomaly Detection**: 0.05-0.3 sensitivity levels
- **🔍 Query Analysis**: 1-14 days of query history
- **🐛 Debug Mode**: Detailed error information

### Advanced Settings
- **Data Caching**: Configurable cache duration
- **Visualization Options**: Chart types and display preferences
- **Export Formats**: CSV with customizable columns
- **Performance Optimization**: Automatic data pagination

## 📈 Key Metrics Explained

### Warehouse Metrics
- **Total Credits**: Compute + cloud services credits consumed
- **Usage Count**: Number of active warehouse periods
- **Credits per Hour**: Efficiency indicator for warehouse utilization
- **Week-over-Week Change**: Percentage change from previous week

### Query Metrics
- **Execution Time**: Total query runtime in seconds
- **Data Scanned**: Bytes processed during query execution
- **Spillage Rate**: Percentage of queries with memory spillage
- **Pruning Effectiveness**: Partition pruning success rate

### Advanced Analytics
- **Anomaly Score**: ML-based outlier likelihood (0-1 scale)
- **Z-Score**: Statistical measure of unusual values
- **Efficiency Rankings**: Performance-based warehouse comparisons

## 🚨 Troubleshooting

### Common Issues

1. **No Data Displayed**
   - ✅ Verify ACCOUNT_USAGE permissions
   - ✅ Check selected date ranges
   - ✅ Confirm warehouse activity in period

2. **Query Analysis Empty**
   - ✅ Verify QUERY_HISTORY access permissions
   - ✅ Check query activity in selected days
   - ✅ Consider data latency (up to 45 minutes)

3. **Performance Issues**
   - ✅ Reduce historical data range
   - ✅ Use warehouse filtering options
   - ✅ Clear cache and refresh data

4. **Feature Unavailability**
   - ✅ Check environment.yaml for missing packages
   - ✅ Verify SiS environment setup
   - ✅ Review feature status in footer

### Debug Mode
Enable debug mode in the sidebar for:
- Detailed error messages
- Data structure insights
- Performance diagnostics
- Feature availability status

## 🔒 Security & Compliance

### Data Security
- **Native Snowflake Environment**: All processing within your Snowflake tenant
- **No External Dependencies**: No data leaves your environment
- **Role-Based Access**: Leverages existing Snowflake RBAC
- **Audit Trail**: All access logged in Snowflake audit logs

### Data Privacy
- **In-Memory Processing**: Temporary caching for performance only
- **No Persistent Storage**: Data not stored outside session
- **Session Isolation**: User sessions completely isolated

## 📊 System Requirements

### Snowflake Requirements
- **Edition**: Standard or higher (for ACCOUNT_USAGE views)
- **Region**: Any supported Snowflake region
- **Warehouse**: Small warehouse sufficient for typical usage

### Streamlit in Snowflake
- **Python Version**: 3.11.* (automatically managed)
- **Dependencies**: Managed via environment.yaml
- **Resource Usage**: Minimal compute requirements

## 🆘 Support & Maintenance

### Getting Help
1. **Check Debug Mode**: Enable for detailed diagnostics
2. **Review Feature Status**: Check footer for package availability
3. **Verify Permissions**: Ensure proper ACCOUNT_USAGE access
4. **Review Documentation**: Use built-in help expander

### Updates & Enhancements
- **Version Control**: Manage code changes through your stage
- **Dependency Updates**: Modify environment.yaml as needed
- **Feature Additions**: Extend functionality with new tabs

## 📝 Version History

### v3.0.0 - Query Analysis Release
- ✨ **NEW**: Comprehensive query-level analysis tab
- ✨ **NEW**: Query performance metrics and efficiency analysis
- ✨ **Enhanced**: Executive summary with custom insight cards
- ✨ **Enhanced**: Improved spike detection with visual cards
- ✨ **Enhanced**: Better UI/UX with professional styling
- 🔧 **Updated**: Streamlit in Snowflake compatibility
- 🔧 **Updated**: Environment.yaml dependency management

### Key Features Added
- Query execution time analysis
- User and role activity tracking  
- Spillage and partition pruning analysis
- Query type distribution insights
- Enhanced anomaly detection with dynamic visualizations
- Professional dashboard styling with color-coded cards

---

**Built for Snowflake ❄️ | Powered by Streamlit 🚀 | Enhanced with ML 🤖**
