import { useQuery } from '@tanstack/react-query'
import { FormEvent, useState } from 'react'
import { api } from '../api'
import { Card, ErrorState, PageHeader } from '../components'

type Track={position:number,track_id:string,label:string,similarity_to_seed:number,similarity_to_previous:number}
type Payload={seed_track_id:string,space:string,tracks:Track[]}
export function Playlists(){
  const [input,setInput]=useState('jazz/jazz.00000'); const [seed,setSeed]=useState(input); const [space,setSpace]=useState('classification')
  const query=useQuery({queryKey:['playlist',seed,space],queryFn:()=>api<Payload>(`/api/playlists/${seed}?space=${space}&length=10`),enabled:!!seed})
  const submit=(e:FormEvent)=>{e.preventDefault();setSeed(input)}
  return <><PageHeader eyebrow="Step 10" title="A playlist that drifts without losing the thread">Walk locally through an embedding while moving gradually farther from the seed. Compare the two learned spaces.</PageHeader>
    <Card><form className="search-form" onSubmit={submit}><label>Seed track<input value={input} onChange={e=>setInput(e.target.value)}/></label><label>Embedding<select value={space} onChange={e=>setSpace(e.target.value)}><option value="classification">Classification-derived</option><option value="triplet">Triplet-loss</option></select></label><button>Generate playlist</button></form></Card>
    {query.error&&<ErrorState error={query.error}/>} {query.data&&<Card><div className="card-heading"><div><span className="eyebrow">Controlled traversal</span><h2>{query.data.seed_track_id}</h2></div><span className="pill">{space}</span></div><div className="playlist-line">{query.data.tracks.map((track,index)=><div className="playlist-item" key={track.track_id}><div className="position">{String(track.position).padStart(2,'0')}</div><div className="playlist-dot"/><div className="playlist-copy"><strong>{track.track_id}</strong><span>{track.label}</span><small>{index===0?'Seed':`${(track.similarity_to_previous*100).toFixed(1)}% similar to previous`}</small></div></div>)}</div></Card>}
    <Card className="disclosure"><strong>No audio is hosted.</strong><p>This first deployment demonstrates model behavior with track identifiers and similarity values. Listening-based review remains a separate, explicitly human step.</p></Card>
  </>
}
