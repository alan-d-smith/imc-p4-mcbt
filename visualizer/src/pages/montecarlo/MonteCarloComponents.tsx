import { Container, Group, Loader, Table, Text } from '@mantine/core';
import Highcharts from 'highcharts';
import HighchartsMore from 'highcharts/highcharts-more';
import HighchartsAccessibility from 'highcharts/modules/accessibility';
import HighchartsExporting from 'highcharts/modules/exporting';
import HighchartsOfflineExporting from 'highcharts/modules/offline-exporting';
import HighchartsReact from 'highcharts-react-official';
import merge from 'lodash/merge';
import { ReactNode, useMemo } from 'react';
import { ErrorAlert } from '../../components/ErrorAlert.tsx';
import { useActualColorScheme } from '../../hooks/use-actual-color-scheme.ts';
import { MonteCarloBandSeries, MonteCarloDashboard, MonteCarloDistributionStats, MonteCarloHistogram, MonteCarloNormalFit } from '../../models.ts';
import { formatNumber } from '../../utils/format.ts';
import { VisualizerCard } from '../visualizer/VisualizerCard.tsx';

HighchartsAccessibility(Highcharts);
HighchartsExporting(Highcharts);
HighchartsOfflineExporting(Highcharts);
HighchartsMore(Highcharts);

export interface SimpleChartProps {
  title: string;
  subtitle?: string;
  series: Highcharts.SeriesOptionsType[];
  options?: Highcharts.Options;
}

export function SimpleChart({ title, subtitle, series, options }: SimpleChartProps): ReactNode {
  const colorScheme = useActualColorScheme();
  const axisTextColor = colorScheme === 'dark' ? '#ffffff' : '#495057';
  const axisLineColor = colorScheme === 'dark' ? '#f1f3f5' : '#ced4da';
  const gridLineColor = colorScheme === 'dark' ? '#6c757d' : '#e9ecef';

  const fullOptions = useMemo(
    (): Highcharts.Options =>
      merge({}, {
        chart: {
          animation: false,
          height: 420,
          backgroundColor: 'transparent',
          plotBackgroundColor: 'transparent',
          numberFormatter: formatNumber,
        },
        title: { text: undefined },
        subtitle: subtitle
          ? {
              text: subtitle,
              style: {
                color: colorScheme === 'dark' ? '#adb5bd' : '#495057',
              },
            }
          : undefined,
        credits: {
          href: 'javascript:window.open("https://www.highcharts.com/?credits", "_blank")',
          style: {
            color: colorScheme === 'dark' ? '#868e96' : '#868e96',
          },
        },
        xAxis: {
          type: 'linear',
          gridLineColor,
          lineColor: axisLineColor,
          tickColor: axisLineColor,
          crosshair: {
            width: 1,
            color: colorScheme === 'dark' ? '#adb5bd' : '#adb5bd',
          },
          title: {
            style: {
              color: axisTextColor,
              textOutline: 'none',
            },
          },
          labels: {
            style: {
              color: axisTextColor,
              fontSize: '13px',
              fontWeight: '600',
              textOutline: 'none',
            },
            formatter(this: Highcharts.AxisLabelsFormatterContextObject) {
              return formatNumber(Number(this.value));
            },
          },
        },
        yAxis: {
          opposite: false,
          gridLineColor,
          lineColor: axisLineColor,
          tickColor: axisLineColor,
          title: {
            style: {
              color: axisTextColor,
              textOutline: 'none',
            },
          },
          labels: {
            style: {
              color: axisTextColor,
              fontSize: '13px',
              fontWeight: '600',
              textOutline: 'none',
            },
            formatter(this: Highcharts.AxisLabelsFormatterContextObject) {
              return formatNumber(Number(this.value));
            },
          },
        },
        tooltip: {
          shared: true,
          outside: true,
          useHTML: true,
          backgroundColor: colorScheme === 'dark' ? '#1f2328' : '#ffffff',
          borderColor: colorScheme === 'dark' ? '#495057' : '#ced4da',
          style: {
            color: colorScheme === 'dark' ? '#f8f9fa' : '#212529',
          },
        },
        legend: {
          enabled: true,
          itemStyle: {
            color: colorScheme === 'dark' ? '#e9ecef' : '#212529',
          },
          itemHoverStyle: {
            color: colorScheme === 'dark' ? '#ffffff' : '#000000',
          },
          itemHiddenStyle: {
            color: colorScheme === 'dark' ? '#6c757d' : '#adb5bd',
          },
        },
        plotOptions: {
          series: {
            animation: false,
            marker: { enabled: false },
            states: { inactive: { opacity: 1 } },
          },
          scatter: {
            marker: {
              enabled: true,
              radius: 3,
            },
          },
        },
        navigator: { enabled: false },
        rangeSelector: { enabled: false },
        scrollbar: { enabled: false },
        exporting: {
          buttons: {
            contextButton: {
              theme: {
                fill: colorScheme === 'dark' ? '#25262b' : '#ffffff',
                stroke: colorScheme === 'dark' ? '#495057' : '#ced4da',
              },
            },
          },
        },
        series,
      }, options ?? {}),
    [axisLineColor, axisTextColor, colorScheme, gridLineColor, options, series, subtitle],
  );

  return (
    <VisualizerCard p={0} title={title}>
      <HighchartsReact highcharts={Highcharts} constructorType="chart" options={fullOptions} immutable />
    </VisualizerCard>
  );
}

export function formatSlope(value: number): string {
  return `$${formatNumber(value, 4)}/step`;
}

export function histogramCenters(histogram: MonteCarloHistogram): Array<[number, number]> {
  return histogram.counts.map((count, index) => [
    (histogram.binEdges[index] + histogram.binEdges[index + 1]) / 2,
    count,
  ]);
}

export function histogramSeries(histogram: MonteCarloHistogram, name: string, color: string): Highcharts.SeriesColumnOptions {
  return {
    type: 'column',
    name,
    color,
    data: histogramCenters(histogram),
  };
}

export function normalFitSeries(fit: MonteCarloNormalFit): Highcharts.SeriesSplineOptions {
  return {
    type: 'spline',
    name: 'Normal fit',
    color: '#fa5252',
    lineWidth: 2,
    data: fit.line,
  };
}

export function distributionLineSeries(histogram: MonteCarloHistogram, name: string, color: string): Highcharts.SeriesLineOptions {
  return {
    type: 'line',
    name,
    color,
    lineWidth: 2,
    data: histogramCenters(histogram),
  };
}

export function bandAreaSeries(
  name: string,
  color: string,
  timestamps: number[],
  lower: number[],
  upper: number[],
  opacity: number,
): Highcharts.SeriesArearangeOptions {
  return {
    type: 'arearange',
    name,
    color,
    fillOpacity: opacity,
    lineColor: color,
    lineWidth: 1,
    data: timestamps.map((timestamp, index) => [timestamp, lower[index], upper[index]]),
    tooltip: {
      pointFormat: `<span style="color:${color}">{series.name}</span>: {point.low:.2f} to {point.high:.2f}<br/>`,
    },
  };
}

export function lineSeries(
  name: string,
  color: string,
  timestamps: number[],
  values: number[],
  dashStyle?: Highcharts.DashStyleValue,
): Highcharts.SeriesLineOptions {
  return {
    type: 'line',
    name,
    color,
    lineWidth: dashStyle ? 1 : 2,
    dashStyle,
    data: timestamps.map((timestamp, index) => [timestamp, values[index]]),
    tooltip: {
      pointFormat: `<span style="color:${color}">{series.name}</span>: {point.y:.2f}<br/>`,
    },
  };
}

export function SummaryTable({ title, stats }: { title: string; stats: MonteCarloDistributionStats }): ReactNode {
  const rows: Array<[string, string]> = [
    ['Mean', formatNumber(stats.mean, 2)],
    ['1σ', formatNumber(stats.std, 2)],
    ['P05', formatNumber(stats.p05, 2)],
    ['Median', formatNumber(stats.p50, 2)],
    ['P95', formatNumber(stats.p95, 2)],
    ['95% Mean CI', `${formatNumber(stats.meanConfidenceLow95, 2)} to ${formatNumber(stats.meanConfidenceHigh95, 2)}`],
  ];

  return (
    <VisualizerCard title={title}>
      <Table striped withTableBorder withColumnBorders>
        <Table.Tbody>
          {rows.map(([label, value]) => (
            <Table.Tr key={`${title}-${label}`}>
              <Table.Td>{label}</Table.Td>
              <Table.Td>{value}</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </VisualizerCard>
  );
}

function productValue(row: any, product: string, legacyKey: string): number {
  return row?.productPnls?.[product] ?? row?.[legacyKey] ?? 0;
}

export function SessionRankingTable({
  title,
  rows,
  productOrder,
  productLabels,
}: {
  title: string;
  rows: MonteCarloDashboard['sessions'];
  productOrder: string[];
  productLabels: Record<string, string>;
}): ReactNode {
  const firstProduct = productOrder[0] ?? 'EMERALDS';
  const secondProduct = productOrder[1] ?? firstProduct;
  const firstLegacyKey = firstProduct === productOrder[0] ? 'emeraldPnl' : 'emeraldPnl';
  const secondLegacyKey = secondProduct === productOrder[1] ? 'tomatoPnl' : 'tomatoPnl';

  return (
    <VisualizerCard title={title}>
      <Table striped withTableBorder withColumnBorders stickyHeader stickyHeaderOffset={0}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Session</Table.Th>
            <Table.Th>Total</Table.Th>
            <Table.Th>{productLabels[firstProduct] ?? firstProduct}</Table.Th>
            <Table.Th>{productLabels[secondProduct] ?? secondProduct}</Table.Th>
            <Table.Th>Total $/step</Table.Th>
            <Table.Th>Total R²</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((row: any) => (
            <Table.Tr key={`${title}-${row.sessionId}`}>
              <Table.Td>{row.sessionId}</Table.Td>
              <Table.Td>{formatNumber(row.totalPnl, 2)}</Table.Td>
              <Table.Td>{formatNumber(productValue(row, firstProduct, firstLegacyKey), 2)}</Table.Td>
              <Table.Td>{formatNumber(productValue(row, secondProduct, secondLegacyKey), 2)}</Table.Td>
              <Table.Td>{formatNumber(row.runMeanTotalSlopePerStep ?? row.totalSlopePerStep, 4)}</Table.Td>
              <Table.Td>{formatNumber(row.runMeanTotalR2 ?? row.totalR2, 3)}</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </VisualizerCard>
  );
}

export function LoadingMonteCarloView({ status }: { status: string }): ReactNode {
  return (
    <Container size="md" py="xl">
      <VisualizerCard title="Loading Monte Carlo dashboard">
        <Group>
          <Loader size="sm" />
          <Text>{status}</Text>
        </Group>
      </VisualizerCard>
    </Container>
  );
}

export function ErrorMonteCarloView({ error }: { error: Error }): ReactNode {
  return (
    <Container size="md" py="xl">
      <VisualizerCard title="Failed to load Monte Carlo dashboard">
        <ErrorAlert error={error} />
      </VisualizerCard>
    </Container>
  );
}

export function buildBandChartSeries(series: MonteCarloBandSeries, color: string): Highcharts.SeriesOptionsType[] {
  return [
    bandAreaSeries('±3σ', color, series.timestamps, series.std3Low, series.std3High, 0.08),
    bandAreaSeries('±1σ', color, series.timestamps, series.std1Low, series.std1High, 0.18),
    lineSeries('Mean', color, series.timestamps, series.mean),
  ];
}
