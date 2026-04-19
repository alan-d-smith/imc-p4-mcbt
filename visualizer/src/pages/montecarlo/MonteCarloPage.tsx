import { Badge, Container, Grid, Group, Select, Table, Text, Title } from '@mantine/core';
import axios from 'axios';
import Highcharts from 'highcharts';
import { ReactNode, useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { MonteCarloDashboard } from '../../models.ts';
import { useStore } from '../../store.ts';
import { parseVisualizerInput } from '../../utils/algorithm.tsx';
import { formatNumber } from '../../utils/format.ts';
import { VisualizerCard } from '../visualizer/VisualizerCard.tsx';
import {
  buildBandChartSeries,
  distributionLineSeries,
  ErrorMonteCarloView,
  formatSlope,
  histogramSeries,
  lineSeries,
  LoadingMonteCarloView,
  normalFitSeries,
  SessionRankingTable,
  SimpleChart,
  SummaryTable,
} from './MonteCarloComponents.tsx';

function basename(path: string): string {
  const normalized = path.replace(/\\/g, '/');
  return normalized.split('/').filter(Boolean).pop() ?? path;
}

function withVersion(url: string, version: string | null): string {
  if (version === null || version.length === 0) {
    return url;
  }

  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}v=${encodeURIComponent(version)}`;
}

function productOrder(dashboard: any): string[] {
  const fromMeta = dashboard?.meta?.productOrder;
  if (Array.isArray(fromMeta) && fromMeta.length > 0) {
    return fromMeta;
  }

  const fromProducts = Object.keys(dashboard?.products ?? {});
  if (fromProducts.length > 0) {
    return fromProducts;
  }

  return ['EMERALDS', 'TOMATOES'];
}

function productLabels(dashboard: any, order: string[]): Record<string, string> {
  const displayNames = dashboard?.meta?.productDisplayNames ?? {};
  return Object.fromEntries(order.map(product => [product, displayNames[product] ?? product]));
}

function productColor(dashboard: any, product: string, fallback: string): string {
  return dashboard?.meta?.productColors?.[product] ?? fallback;
}

function productTrend(dashboard: any, product: string, legacyKey: 'EMERALDS' | 'TOMATOES'): any {
  return dashboard?.trendFits?.[product] ?? dashboard?.trendFits?.[legacyKey];
}

function productHistogram(dashboard: any, product: string, section: string, legacyKey: string): any {
  return dashboard?.histograms?.[section]?.[product] ?? dashboard?.histograms?.[legacyKey];
}

function productNormalFit(dashboard: any, product: string, legacyKey: string): any {
  return dashboard?.normalFits?.productPnl?.[product] ?? dashboard?.normalFits?.[legacyKey];
}

function productSummary(dashboard: any, product: string): any {
  return dashboard?.products?.[product]?.pnl;
}

function sessionProductPnl(row: any, product: string, legacyKey: string): number {
  return row?.productPnls?.[product] ?? row?.[legacyKey] ?? 0;
}

type LocalDashboardStatus = {
  dashboardExists: boolean;
  dashboardMtimeMs: number | null;
  dashboardSizeBytes: number | null;
  root: string;
  currentRunId?: string | null;
  runs?: Array<{
    id: string;
    label: string;
    mtimeMs: number;
    dashboardUrl: string;
  }>;
};

export function MonteCarloPage(): ReactNode {
  const storedDashboard = useStore(state => state.monteCarlo);
  const { search } = useLocation();
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [status, setStatus] = useState('Loading Monte Carlo dashboard');
  const [localDashboard, setLocalDashboard] = useState<MonteCarloDashboard | null>(null);
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  const [dashboardVersion, setDashboardVersion] = useState<string | null>(null);
  const [availableRuns, setAvailableRuns] = useState<
    Array<{
      id: string;
      label: string;
      mtimeMs: number;
      dashboardUrl: string;
    }>
  >([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('latest');
  const [bandProduct, setBandProduct] = useState('');
  const searchParams = new URLSearchParams(search);
  const explicitOpenUrl = searchParams.get('open');
  const localMode = typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname);
  const latestRun = availableRuns[0] ?? null;
  const selectedRun =
    selectedRunId === 'latest'
      ? latestRun
      : availableRuns.find(run => run.id === selectedRunId) ?? latestRun;
  const localFallbackOpenUrl = localMode ? selectedRun?.dashboardUrl ?? '/dashboard.json' : null;
  const localStatusUrl = localMode ? '/__prosperity4mcbt__/status.json' : null;
  const openUrl = explicitOpenUrl ?? localFallbackOpenUrl;
  const effectiveOpenUrl = openUrl === null ? null : withVersion(openUrl, explicitOpenUrl === null ? dashboardVersion : null);
  const dashboard = effectiveOpenUrl === null ? storedDashboard : loadedUrl === effectiveOpenUrl ? localDashboard : null;

  useEffect(() => {
    if (localStatusUrl === null || explicitOpenUrl !== null) {
      return;
    }

    let cancelled = false;
    let previousVersion: string | null = null;

    const poll = async (): Promise<void> => {
      try {
        const response = await axios.get<LocalDashboardStatus>(localStatusUrl, {
          headers: {
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
        });

        if (cancelled) {
          return;
        }

        const runs = response.data.runs ?? [];
        setAvailableRuns(runs);

        if (runs.length === 0) {
          setSelectedRunId('latest');
        } else {
          setSelectedRunId(previous => {
            if (previous === 'latest') {
              return 'latest';
            }
            return runs.some(run => run.id === previous) ? previous : 'latest';
          });
        }

        if (!response.data.dashboardExists || response.data.dashboardMtimeMs === null) {
          setLocalDashboard(null);
          setLoadedUrl(null);
          setDashboardVersion(null);
          return;
        }

        const currentRun =
          response.data.currentRunId === undefined || response.data.currentRunId === null
            ? runs[0]
            : runs.find(run => run.id === response.data.currentRunId) ?? runs[0];
        const selectedForVersion = selectedRunId === 'latest' ? currentRun : runs.find(run => run.id === selectedRunId) ?? currentRun;
        const nextVersion = selectedForVersion === undefined ? String(response.data.dashboardMtimeMs) : String(selectedForVersion.mtimeMs);
        if (previousVersion !== nextVersion) {
          previousVersion = nextVersion;
          setDashboardVersion(nextVersion);
        }
      } catch {
        if (!cancelled) {
          setStatus('Waiting for local dashboard');
        }
      }
    };

    void poll();
    const interval = window.setInterval(() => {
      void poll();
    }, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [explicitOpenUrl, localStatusUrl, selectedRunId]);

  useEffect(() => {
    if (effectiveOpenUrl === null) {
      setLoadError(null);
      setStatus('Loading Monte Carlo dashboard');
      setLocalDashboard(null);
      setLoadedUrl(null);
      return;
    }

    if (effectiveOpenUrl.trim().length === 0) {
      return;
    }

    let cancelled = false;
    setLoadError(null);
    setStatus('Fetching dashboard');
    setLocalDashboard(null);
    setLoadedUrl(null);
    const load = async (): Promise<void> => {
      try {
        const response = await axios.get(effectiveOpenUrl, {
          headers: {
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
        });
        const parsed = parseVisualizerInput(response.data);

        if (cancelled) {
          return;
        }

        if (parsed.kind === 'monteCarlo') {
          setLocalDashboard(parsed.monteCarlo);
          setLoadedUrl(effectiveOpenUrl);
          setStatus('Dashboard loaded');
          return;
        }

        setLoadError(new Error('This visualizer build only supports Monte Carlo dashboard bundles.'));
      } catch (error) {
        if (cancelled) {
          return;
        }
        setLoadError(error as Error);
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, [effectiveOpenUrl]);

  useEffect(() => {
    const order = dashboard ? productOrder(dashboard as any) : [];
    if (bandProduct !== '' && dashboard?.bandSeries?.[bandProduct] !== undefined) {
      return;
    }
    const fallback = order[1] ?? order[0] ?? '';
    if (fallback !== '') {
      setBandProduct(fallback);
    }
  }, [bandProduct, dashboard]);

  if (dashboard === null) {
    if (loadError !== null) {
      return <ErrorMonteCarloView error={loadError} />;
    }

    if (openUrl !== null) {
      return <LoadingMonteCarloView status={status} />;
    }
    return <LoadingMonteCarloView status={status} />;
  }

  const dashboardAny: any = dashboard;
  const strategyName = basename(dashboard.meta.algorithmPath);
  const order = productOrder(dashboardAny);
  const labels = productLabels(dashboardAny, order);
  const primaryProduct = order[0] ?? 'EMERALDS';
  const secondaryProduct = order[1] ?? primaryProduct;
  const primaryLegacyKey = 'emeraldPnl';
  const secondaryLegacyKey = 'tomatoPnl';
  const primaryColor = productColor(dashboardAny, primaryProduct, '#12b886');
  const secondaryColor = productColor(dashboardAny, secondaryProduct, '#fd7e14');
  const totalTrend = dashboard.trendFits.TOTAL;
  const primaryTrend = productTrend(dashboardAny, primaryProduct, 'EMERALDS');
  const secondaryTrend = productTrend(dashboardAny, secondaryProduct, 'TOMATOES');
  const scatterFit = dashboard.scatterFit;
  const selectedBandSeries = dashboard.bandSeries?.[bandProduct];
  const bandOptions = order.map(product => ({ value: product, label: labels[product] ?? product }));

  const totalHistogramSeries: Highcharts.SeriesOptionsType[] = [
    histogramSeries(dashboard.histograms.totalPnl, 'Total PnL', '#4c6ef5'),
    normalFitSeries(dashboard.normalFits.totalPnl),
  ];
  const primaryHistogramSeries: Highcharts.SeriesOptionsType[] = [
    histogramSeries(productHistogram(dashboardAny, primaryProduct, 'productPnl', primaryLegacyKey), `${labels[primaryProduct]} PnL`, primaryColor),
    normalFitSeries(productNormalFit(dashboardAny, primaryProduct, primaryLegacyKey)),
  ];
  const secondaryHistogramSeries: Highcharts.SeriesOptionsType[] = [
    histogramSeries(productHistogram(dashboardAny, secondaryProduct, 'productPnl', secondaryLegacyKey), `${labels[secondaryProduct]} PnL`, secondaryColor),
    normalFitSeries(productNormalFit(dashboardAny, secondaryProduct, secondaryLegacyKey)),
  ];
  const scatterSeries: Highcharts.SeriesOptionsType[] = [
    {
      type: 'scatter',
      name: 'Sessions',
      color: '#4c6ef5',
      data: dashboard.sessions.map((row: any) => [
        sessionProductPnl(row, primaryProduct, primaryLegacyKey),
        sessionProductPnl(row, secondaryProduct, secondaryLegacyKey),
      ]),
    },
    {
      type: 'line',
      name: 'Linear fit',
      color: '#fa5252',
      lineWidth: 2,
      data: scatterFit.line,
    },
  ];
  const profitabilitySeries: Highcharts.SeriesOptionsType[] = [
    distributionLineSeries(dashboard.histograms.totalProfitability, 'Total', '#4c6ef5'),
    ...order.map(product =>
      distributionLineSeries(
        productHistogram(dashboardAny, product, 'productProfitability', product === primaryProduct ? 'emeraldProfitability' : 'tomatoProfitability'),
        labels[product] ?? product,
        productColor(dashboardAny, product, product === primaryProduct ? '#12b886' : '#fd7e14'),
      ),
    ),
  ];
  const stabilitySeries: Highcharts.SeriesOptionsType[] = [
    distributionLineSeries(dashboard.histograms.totalStability, 'Total', '#4c6ef5'),
    ...order.map(product =>
      distributionLineSeries(
        productHistogram(dashboardAny, product, 'productStability', product === primaryProduct ? 'emeraldStability' : 'tomatoStability'),
        labels[product] ?? product,
        productColor(dashboardAny, product, product === primaryProduct ? '#12b886' : '#fd7e14'),
      ),
    ),
  ];

  return (
    <Container fluid py="md">
      <Grid>
        <Grid.Col span={12}>
          <VisualizerCard>
            <Group justify="space-between" align="flex-start">
              <div>
                <Title order={2}>Monte Carlo Results</Title>
                <Text c="dimmed">{strategyName}</Text>
              </div>
              <Group gap="xs" align="flex-start">
                {explicitOpenUrl === null && localMode && availableRuns.length > 0 && (
                  <Select
                    w={260}
                    label="Run"
                    value={selectedRunId}
                    onChange={value => setSelectedRunId(value ?? 'latest')}
                    allowDeselect={false}
                    data={[
                      {
                        value: 'latest',
                        label: `Latest (${latestRun?.label ?? 'none'})`,
                      },
                      ...availableRuns.map(run => ({
                        value: run.id,
                        label: run.label,
                      })),
                    ]}
                  />
                )}
                {dashboardAny.meta.round !== undefined && <Badge variant="light">Round {dashboardAny.meta.round}</Badge>}
                <Badge variant="light">{dashboard.meta.sessionCount} sessions</Badge>
                <Badge variant="light">{dashboard.meta.bandSessionCount ?? dashboard.meta.sampleSessions} path traces</Badge>
                {dashboardAny.meta.blockSize !== undefined && <Badge variant="light">block {dashboardAny.meta.blockSize}</Badge>}
                {dashboardAny.meta.simulationModel !== undefined && <Badge variant="light">{dashboardAny.meta.simulationModel}</Badge>}
              </Group>
            </Group>
          </VisualizerCard>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 6 }}>
          <VisualizerCard title="Mean Total PnL">
            <Title order={2}>{formatNumber(dashboard.overall.totalPnl.mean)}</Title>
            <Text c="dimmed" size="sm">
              95% mean CI {formatNumber(dashboard.overall.totalPnl.meanConfidenceLow95)} to{' '}
              {formatNumber(dashboard.overall.totalPnl.meanConfidenceHigh95)}
            </Text>
          </VisualizerCard>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <VisualizerCard title="Total PnL 1σ">
            <Title order={2}>{formatNumber(dashboard.overall.totalPnl.std)}</Title>
            <Text c="dimmed" size="sm">
              P05 {formatNumber(dashboard.overall.totalPnl.p05)} · P95 {formatNumber(dashboard.overall.totalPnl.p95)}
            </Text>
          </VisualizerCard>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 8 }}>
          <VisualizerCard title="Profitability And Statistics">
            <Table striped withTableBorder withColumnBorders>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Metric</Table.Th>
                  <Table.Th>Meaning</Table.Th>
                  <Table.Th>Total</Table.Th>
                  <Table.Th>{labels[primaryProduct] ?? primaryProduct}</Table.Th>
                  <Table.Th>{labels[secondaryProduct] ?? secondaryProduct}</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                <Table.Tr>
                  <Table.Td>Profitability</Table.Td>
                  <Table.Td>Mean fitted MTM slope in dollars per step.</Table.Td>
                  <Table.Td>{formatSlope(totalTrend.profitability.mean)}</Table.Td>
                  <Table.Td>{formatSlope(primaryTrend.profitability.mean)}</Table.Td>
                  <Table.Td>{formatSlope(secondaryTrend.profitability.mean)}</Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Stability</Table.Td>
                  <Table.Td>Mean linear-fit R². Higher means steadier PnL paths.</Table.Td>
                  <Table.Td>{formatNumber(totalTrend.stability.mean, 3)}</Table.Td>
                  <Table.Td>{formatNumber(primaryTrend.stability.mean, 3)}</Table.Td>
                  <Table.Td>{formatNumber(secondaryTrend.stability.mean, 3)}</Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Profitability 1σ</Table.Td>
                  <Table.Td>Cross-session spread of profitability.</Table.Td>
                  <Table.Td>{formatSlope(totalTrend.profitability.std)}</Table.Td>
                  <Table.Td>{formatSlope(primaryTrend.profitability.std)}</Table.Td>
                  <Table.Td>{formatSlope(secondaryTrend.profitability.std)}</Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Stability 1σ</Table.Td>
                  <Table.Td>Cross-session spread of stability.</Table.Td>
                  <Table.Td>{formatNumber(totalTrend.stability.std, 3)}</Table.Td>
                  <Table.Td>{formatNumber(primaryTrend.stability.std, 3)}</Table.Td>
                  <Table.Td>{formatNumber(secondaryTrend.stability.std, 3)}</Table.Td>
                </Table.Tr>
              </Table.Tbody>
            </Table>
          </VisualizerCard>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 4 }}>
          <VisualizerCard title="Market Model">
            <Table withTableBorder withColumnBorders>
              <Table.Tbody>
                {order.map(product => (
                  <Table.Tr key={product}>
                    <Table.Td>{labels[product] ?? product}</Table.Td>
                    <Table.Td>
                      <Text fw={500}>{dashboard.generatorModel[product]?.formula ?? 'N/A'}</Text>
                      <Text size="sm" c="dimmed">
                        {dashboard.generatorModel[product]?.notes?.[0] ?? 'No model notes available'}
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </VisualizerCard>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 4 }}>
          <SummaryTable title="Total PnL Summary" stats={dashboard.overall.totalPnl} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 4 }}>
          <SummaryTable title={`${labels[primaryProduct] ?? primaryProduct} PnL Summary`} stats={productSummary(dashboardAny, primaryProduct)} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 4 }}>
          <SummaryTable title={`${labels[secondaryProduct] ?? secondaryProduct} PnL Summary`} stats={productSummary(dashboardAny, secondaryProduct)} />
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 6 }}>
          <SimpleChart
            title="Total PnL Distribution"
            subtitle={`Normal fit μ ${formatNumber(dashboard.normalFits.totalPnl.mean)} · σ ${formatNumber(dashboard.normalFits.totalPnl.std)} · R² ${formatNumber(dashboard.normalFits.totalPnl.r2, 3)}`}
            series={totalHistogramSeries}
            options={{
              xAxis: { title: { text: 'Final total pnl' } },
              yAxis: { title: { text: 'Session count' } },
            }}
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <SimpleChart
            title="Cross Product Scatter"
            subtitle={`corr ${formatNumber(scatterFit.correlation, 3)} · fit R² ${formatNumber(scatterFit.r2, 3)} · ${scatterFit.diagnosis}`}
            series={scatterSeries}
            options={{
              xAxis: { title: { text: `${labels[primaryProduct] ?? primaryProduct} pnl` } },
              yAxis: { title: { text: `${labels[secondaryProduct] ?? secondaryProduct} pnl` } },
            }}
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <SimpleChart
            title={`${labels[primaryProduct] ?? primaryProduct} PnL Distribution`}
            subtitle={`Normal fit μ ${formatNumber(productNormalFit(dashboardAny, primaryProduct, primaryLegacyKey).mean)} · σ ${formatNumber(productNormalFit(dashboardAny, primaryProduct, primaryLegacyKey).std)} · R² ${formatNumber(productNormalFit(dashboardAny, primaryProduct, primaryLegacyKey).r2, 3)}`}
            series={primaryHistogramSeries}
            options={{
              xAxis: { title: { text: `${labels[primaryProduct] ?? primaryProduct} final pnl` } },
              yAxis: { title: { text: 'Session count' } },
            }}
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <SimpleChart
            title={`${labels[secondaryProduct] ?? secondaryProduct} PnL Distribution`}
            subtitle={`Normal fit μ ${formatNumber(productNormalFit(dashboardAny, secondaryProduct, secondaryLegacyKey).mean)} · σ ${formatNumber(productNormalFit(dashboardAny, secondaryProduct, secondaryLegacyKey).std)} · R² ${formatNumber(productNormalFit(dashboardAny, secondaryProduct, secondaryLegacyKey).r2, 3)}`}
            series={secondaryHistogramSeries}
            options={{
              xAxis: { title: { text: `${labels[secondaryProduct] ?? secondaryProduct} final pnl` } },
              yAxis: { title: { text: 'Session count' } },
            }}
          />
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 6 }}>
          <SimpleChart
            title="Profitability Distribution"
            subtitle="Per-session fitted MTM slope in dollars per step"
            series={profitabilitySeries}
            options={{
              xAxis: {
                title: { text: '$ / step' },
                labels: {
                  formatter(this: Highcharts.AxisLabelsFormatterContextObject) {
                    return formatNumber(Number(this.value), 4);
                  },
                },
              },
              yAxis: { title: { text: 'Density proxy' } },
            }}
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <SimpleChart
            title="Stability Distribution"
            subtitle="Per-session linear-fit R²"
            series={stabilitySeries}
            options={{
              xAxis: {
                title: { text: 'R²' },
                labels: {
                  formatter(this: Highcharts.AxisLabelsFormatterContextObject) {
                    return formatNumber(Number(this.value), 3);
                  },
                },
              },
              yAxis: { title: { text: 'Density proxy' } },
            }}
          />
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 6 }}>
          <SessionRankingTable title="Best Sessions" rows={dashboard.topSessions} productOrder={order} productLabels={labels} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <SessionRankingTable title="Worst Sessions" rows={dashboard.bottomSessions} productOrder={order} productLabels={labels} />
        </Grid.Col>

        {selectedBandSeries && (
          <>
            <Grid.Col span={12}>
              <VisualizerCard title="Path Boards">
                <Group justify="space-between" align="center">
                  <Text c="dimmed" size="sm">
                    Mean path with ±1σ and ±3σ bands across {dashboard.meta.bandSessionCount ?? dashboard.meta.sampleSessions} sessions.
                  </Text>
                  <Select
                    w={220}
                    data={bandOptions}
                    value={bandProduct}
                    onChange={value => setBandProduct(value ?? primaryProduct)}
                    allowDeselect={false}
                  />
                </Group>
              </VisualizerCard>
            </Grid.Col>
            <Grid.Col span={12}>
              <SimpleChart
                title={`${labels[bandProduct] ?? bandProduct} Mid Price Proxy`}
                series={buildBandChartSeries(selectedBandSeries.fair, productColor(dashboardAny, bandProduct, bandProduct === primaryProduct ? primaryColor : secondaryColor))}
                options={{
                  xAxis: {
                    title: { text: 'Step' },
                  },
                  yAxis: { title: { text: 'Price' } },
                }}
              />
            </Grid.Col>
            <Grid.Col span={12}>
              <SimpleChart
                title={`${labels[bandProduct] ?? bandProduct} MTM PnL`}
                series={[
                  ...buildBandChartSeries(selectedBandSeries.mtmPnl, productColor(dashboardAny, bandProduct, bandProduct === primaryProduct ? primaryColor : secondaryColor)),
                  lineSeries('Zero', '#868e96', selectedBandSeries.mtmPnl.timestamps, selectedBandSeries.mtmPnl.timestamps.map(() => 0), 'ShortDash'),
                ]}
                options={{
                  xAxis: {
                    title: { text: 'Step' },
                  },
                  yAxis: { title: { text: 'MTM pnl' } },
                }}
              />
            </Grid.Col>
            <Grid.Col span={12}>
              <SimpleChart
                title={`${labels[bandProduct] ?? bandProduct} Position`}
                series={[
                  ...buildBandChartSeries(selectedBandSeries.position, productColor(dashboardAny, bandProduct, bandProduct === primaryProduct ? primaryColor : secondaryColor)),
                  lineSeries('Zero', '#868e96', selectedBandSeries.position.timestamps, selectedBandSeries.position.timestamps.map(() => 0), 'ShortDash'),
                ]}
                options={{
                  xAxis: {
                    title: { text: 'Step' },
                  },
                  yAxis: { title: { text: 'Position' } },
                }}
              />
            </Grid.Col>
          </>
        )}
      </Grid>
    </Container>
  );
}
