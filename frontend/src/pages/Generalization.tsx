import { useQuery } from '@tanstack/react-query'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { Card, ErrorState, Loading, Metric, PageHeader, fmt } from '../components'

type Payload = {classification:Array<Record<string,number>>,per_genre:Array<{genre:string,accuracy:number,n:number}>,similarity:Array<Record<string,number|string>>}
export function Generalization(){
  const query=useQuery({queryKey:['generalization'],queryFn:()=>api<Payload>('/api/generalization')})
  if(query.isLoading)return <Loading/>; if(query.error)return <ErrorState error={query.error}/>; const data=query.data!; const c=data.classification[0]
  const k10=data.similarity.filter(row=>row.k===10)
  return <><PageHeader eyebrow="Step 9" title="The distribution shift was not subtle">Models trained on GTZAN met 2,999 unseen FMA tracks from the three defensible exact-overlap genres.</PageHeader>
    <div className="metrics-grid"><Metric label="GTZAN CV accuracy" value={`${fmt(c.gtzan_cv_accuracy*100,1)}%`} note="Augmented CNN"/><Metric label="FMA accuracy" value={`${fmt(c.accuracy*100,1)}%`} note="95% CI 33.0–36.2%"/><Metric label="Absolute drop" value={`${fmt(c.absolute_accuracy_drop*100,1)} pts`} note="A genuine generalization gap"/><Metric label="FMA macro F1" value={fmt(c.f1_macro)} note="Pop was hardest"/></div>
    <div className="two-col"><Card><div className="card-heading"><h2>Accuracy by FMA genre</h2><span className="pill">No retraining</span></div><ResponsiveContainer width="100%" height={300}><BarChart data={data.per_genre}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="genre"/><YAxis domain={[0,.6]}/><Tooltip/><Bar dataKey="accuracy" fill="#ee7183" radius={[6,6,0,0]}/></BarChart></ResponsiveContainer></Card>
      <Card><div className="card-heading"><h2>Embedding robustness at k=10</h2><span className="pill">Chance-adjusted</span></div>{k10.map(row=><div className="comparison-row" key={String(row.method)}><div><strong>{String(row.method).replace('_embedding','')}</strong><span>FMA P@10 {Number(row.fma_precision_at_k).toFixed(3)}</span></div><div className="bar-track"><i style={{width:`${Number(row.adjusted_precision_drop)*100}%`}}/></div><b>−{Number(row.adjusted_precision_drop).toFixed(3)}</b></div>)}<p className="annotation">The triplet embedding was not the strongest absolute retriever, but its chance-adjusted degradation was smaller.</p></Card></div>
  </>
}
