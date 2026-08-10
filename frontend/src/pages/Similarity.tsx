import { useQuery } from '@tanstack/react-query'
import { FormEvent, useState } from 'react'
import { api } from '../api'
import { Card, ErrorState, PageHeader } from '../components'

type Neighbor = {rank:number,neighbor_track_id:string,neighbor_label:string,distance:number,same_genre:boolean}
type Payload = {neighbors:Neighbor[],method:string}
const methods = ['classification_embedding','triplet_embedding','engineered_features','metadata_genre']

export function Similarity() {
  const [input,setInput] = useState('blues/blues.00000')
  const [track,setTrack] = useState(input)
  const [method,setMethod] = useState(methods[0])
  const query = useQuery({queryKey:['similarity',track,method],queryFn:()=>api<Payload>(`/api/similarity/${track}?method=${method}&k=10`),enabled:!!track})
  const submit=(event:FormEvent)=>{event.preventDefault();setTrack(input)}
  return <><PageHeader eyebrow="Step 8" title="Ask the catalog what sounds close">Compare the same seed across learned embeddings, engineered audio descriptors, and the label-only oracle baseline.</PageHeader>
    <Card><form className="search-form" onSubmit={submit}><label>Seed track<input value={input} onChange={e=>setInput(e.target.value)} placeholder="genre/genre.00000"/></label><label>Representation<select value={method} onChange={e=>setMethod(e.target.value)}>{methods.map(value=><option value={value} key={value}>{value.replaceAll('_',' ')}</option>)}</select></label><button type="submit">Find neighbors</button></form></Card>
    {query.error&&<ErrorState error={query.error}/>} {query.data&&<Card><div className="card-heading"><div><span className="eyebrow">Top 10</span><h2>{track}</h2></div><span className="pill">{method.replaceAll('_',' ')}</span></div><div className="track-list">{query.data.neighbors.map(row=><div className="track-row" key={row.rank}><strong>{String(row.rank).padStart(2,'0')}</strong><div><span>{row.neighbor_track_id}</span><small>{row.neighbor_label}</small></div><span className={row.same_genre?'match':'mismatch'}>{row.same_genre?'same genre':'cross genre'}</span><code>{row.distance.toFixed(4)}</code></div>)}</div></Card>}
    <Card className="disclosure"><strong>Evaluation boundary.</strong><p>Genre agreement is useful for repeatable P@k, but it cannot establish perceptual similarity. The metadata view is deliberately an oracle-like comparison.</p></Card>
  </>
}
