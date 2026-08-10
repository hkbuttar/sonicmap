import { useQuery } from '@tanstack/react-query'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-gl2d-dist-min'
import { useState } from 'react'
import { api, type EmbeddingPoint } from '../api'
import { Card, ErrorState, Loading, PageHeader, fmt } from '../components'

type Payload = {space: string, metrics: Record<string, number>, points: EmbeddingPoint[]}
const palette = ['#8df0c8','#f2bb66','#ee7183','#79b8ff','#c39bf3','#ec8f5e','#8ed081','#f4d35e','#6dd3ce','#d98cb3']
const Plot = createPlotlyComponent(Plotly)

export function Embeddings() {
  const [space,setSpace] = useState<'classification'|'triplet'>('classification')
  const query = useQuery({queryKey:['embedding',space], queryFn:() => api<Payload>(`/api/embeddings/${space}`)})
  if (query.isLoading) return <Loading/>
  if (query.error) return <ErrorState error={query.error}/>
  const data = query.data!
  const genres = [...new Set(data.points.map(point => point.label))]
  const traces = genres.map((genre,index) => {
    const points = data.points.filter(point => point.label === genre)
    return {type:'scattergl' as const, mode:'markers' as const, name:genre, x:points.map(p=>p.x), y:points.map(p=>p.y), text:points.map(p=>p.track_id), hovertemplate:'%{text}<extra>'+genre+'</extra>', marker:{size:6,color:palette[index],opacity:.72}}
  })
  return <><PageHeader eyebrow="Steps 6–7" title="Explore the learned spaces">Toggle between a genre-trained representation and an embedding optimized directly with triplet loss.</PageHeader>
    <div className="toolbar"><div className="segmented"><button className={space==='classification'?'selected':''} onClick={()=>setSpace('classification')}>Classification-derived</button><button className={space==='triplet'?'selected':''} onClick={()=>setSpace('triplet')}>Triplet-loss</button></div><span className="pill">999 tracks · 128 dimensions</span></div>
    <Card className="plot-card"><Plot data={traces} layout={{autosize:true,height:610,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#aeb6c4'},margin:{l:40,r:20,t:20,b:40},xaxis:{gridcolor:'#252b34',zeroline:false,title:{text:'UMAP 1'}},yaxis:{gridcolor:'#252b34',zeroline:false,title:{text:'UMAP 2'}},legend:{orientation:'h',y:-.12}}} config={{responsive:true,displaylogo:false}} style={{width:'100%'}}/></Card>
    <div className="metrics-grid compact"><Card className="metric"><span>Cosine silhouette</span><strong>{fmt(data.metrics.embedding_silhouette_cosine)}</strong><small>Genre separation in 128D</small></Card><Card className="metric"><span>Projection silhouette</span><strong>{fmt(data.metrics.projection_silhouette_euclidean)}</strong><small>Qualitative UMAP check</small></Card>{space==='triplet'&&<Card className="metric"><span>Loss reduction</span><strong>{fmt(data.metrics.initial_triplet_loss-data.metrics.final_triplet_loss)}</strong><small>Objective learned, retrieval still weaker</small></Card>}</div>
  </>
}
