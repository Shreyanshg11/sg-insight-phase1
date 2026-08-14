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
 *
 * `columns` + `onDrillDown` (session 12, Interactive Investigation): when
 * provided, clicking a bar or box-plot group adds an `eq` filter on the
 * category column it represents. Only wired for bar_chart/box_plot in this
 * pass — histogram bins and scatter points don't map onto a single-column
 * `eq` filter as cleanly, so they're left as a fast-follow rather than
 * bolted on awkwardly here.
 */
export default function ChartRenderer({ chartType, data, trend, columns, onDrillDown }) {
  switch (chartType) {
    case "histogram":
      return <BarChart bars={data.bins} />;
    case "bar_chart":
      return (
        <BarChart
          bars={data.categories.map((c) => ({ label: c.value, count: c.count }))}
          onBarClick={onDrillDown && columns?.[0] ? (bar) => onDrillDown({ column: columns[0], op: "eq", value: bar.label }) : undefined}
        />
      );
    case "line_chart":
      return <LineChartView buckets={data.buckets} trend={trend} />;
    case "scatter_plot":
      return <ScatterPlotView points={data.points} />;
    case "box_plot":
      return (
        <BoxPlotView
          groups={data.groups}
          onGroupClick={onDrillDown && columns?.[0] ? (g) => onDrillDown({ column: columns[0], op: "eq", value: g.category }) : undefined}
        />
      );
    case "heatmap":
      return <HeatmapView columns={data.columns} matrix={data.matrix} />;
    default:
      return <p className="font-mono text-xs text-signal-red">Unknown chart type: {chartType}</p>;
  }
}
