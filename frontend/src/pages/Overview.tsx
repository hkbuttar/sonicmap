import { useQuery } from '@tanstack/react-query'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type MetricRow } from '../api'
import { Card, ErrorState, Loading, Metric, PageHeader, fmt } from '../components'

type Summary = { headline: Record<string, number>, disclosure: string }
type Validation = { counts: Record<string, number> }

export function Overview() {
  const summary = useQuery({queryKey:['summary'], queryFn:() => api<Summary>('/api/summary')})
  const validation = useQuery({queryKey:['validation'], queryFn:() => api<Validation>('/api/validation')})
  const genre = useQuery({queryKey:['classification'], queryFn:() => api<MetricRow[]>('/api/classification')})
  if (summary.isLoading || genre.isLoading) return <Loading/>
  if (summary.error || genre.error) return <ErrorState error={(summary.error || genre.error) as Error}/>
  const h = summary.data!.headline
  const accuracy = genre.data!.filter(row => row.metric === 'accuracy').map(row => ({name:`${row.model?.toUpperCase()}${row.augmented ? ' + aug' : ''}`, accuracy:row.mean, low:row.ci_low, high:row.ci_high}))
  return <>
    <PageHeader eyebrow="Audio-native music intelligence" title="What does music sound like to a model?">A rigorous comparison of genre and mood prediction, learned similarity, distribution shift, and playlist coherence.</PageHeader>
    <div className="metrics-grid">
      <Metric label="Best genre accuracy" value={`${fmt(h.genre_accuracy * 100, 1)}%`} note="Augmented CNN · 5-fold CV"/>
      <Metric label="Mood arousal R²" value={fmt(h.mood_arousal_r2)} note="Gradient boosting · DEAM"/>
      <Metric label="Similarity P@10" value={fmt(h.classification_precision_at_10)} note="Classification embedding"/>
      <Metric label="FMA accuracy" value={`${fmt(h.fma_accuracy * 100, 1)}%`} note="Unseen distribution · 3 genres"/>
    </div>
    <div className="two-col">
      <Card><div className="card-heading"><div><span className="eyebrow">Step 4</span><h2>Genre classification</h2></div><span className="pill">95% CI</span></div>
        <ResponsiveContainer width="100%" height={300}><BarChart data={accuracy}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name"/><YAxis domain={[0,1]}/><Tooltip/><Bar dataKey="accuracy" fill="#8df0c8" radius={[6,6,0,0]}/></BarChart></ResponsiveContainer>
      </Card>
      <Card className="finding"><span className="eyebrow">Central finding</span><h2>Simpler won—until distribution shifted.</h2><p>The classification-derived embedding beat triplet learning in-distribution. The triplet space remained weaker on FMA, but lost less quality after adjusting for the different class counts.</p><div className="validation-line"><i className="good"/><strong>{validation.data?.counts.pass ?? 0} validation checks passed</strong><span>· {validation.data?.counts.inconclusive ?? 0} inconclusive · 0 failed</span></div></Card>
    </div>
    <Card className="disclosure"><strong>Read this first.</strong><p>{summary.data!.disclosure} The metadata baseline directly uses the evaluation label and is not audio retrieval.</p></Card>
  </>
}
