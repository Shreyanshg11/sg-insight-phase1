import BarChart from "../eda/BarChart";
import LineChartView from "./LineChartView";
import ScatterPlotView from "./ScatterPlotView";
import BoxPlotView from "./BoxPlotView";
import HeatmapView from "./HeatmapView";

/**
 * Single dispatch point for rendering a chart payload shaped like
 * ChartRecommendation/DashboardChart (chart_type + data + trend). Used by
 * both the Visualization workspace (recommendations) and the Dashboard
 * Builder (pinned charts) so the two never render the same chart_type two
 * different ways.
 */
export default function ChartRenderer({ chartType, data, trend }) {
  switch (chartType) {
    case "histogram":
      return <BarChart bars={data.bins} />;
    case "bar_chart":
      return <BarChart bars={data.categories.map((c) => ({ label: c.value, count: c.count }))} />;
    case "line_chart":
      return <LineChartView buckets={data.buckets} trend={trend} />;
    case "scatter_plot":
      return <ScatterPlotView points={data.points} />;
    case "box_plot":
      return <BoxPlotView groups={data.groups} />;
    case "heatmap":
      return <HeatmapView columns={data.columns} matrix={data.matrix} />;
    default:
      return <p className="font-mono text-xs text-signal-red">Unknown chart type: {chartType}</p>;
  }
}
