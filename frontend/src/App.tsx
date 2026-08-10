import { NavLink, Route, Routes } from 'react-router-dom'
import { Activity, CircleDot, GitCompareArrows, Headphones, Map, Menu, X } from 'lucide-react'
import { lazy, Suspense, useState } from 'react'
import { Loading } from './components'

const Embeddings = lazy(() => import('./pages/Embeddings').then(module => ({default: module.Embeddings})))
const Generalization = lazy(() => import('./pages/Generalization').then(module => ({default: module.Generalization})))
const Overview = lazy(() => import('./pages/Overview').then(module => ({default: module.Overview})))
const Playlists = lazy(() => import('./pages/Playlists').then(module => ({default: module.Playlists})))
const Similarity = lazy(() => import('./pages/Similarity').then(module => ({default: module.Similarity})))

const links = [
  ['/', 'Overview', Activity],
  ['/embeddings', 'Embeddings', Map],
  ['/similarity', 'Similarity', CircleDot],
  ['/generalization', 'Generalization', GitCompareArrows],
  ['/playlists', 'Playlists', Headphones],
] as const

export default function App() {
  const [open, setOpen] = useState(false)
  return <div className="shell">
    <header className="topbar">
      <NavLink to="/" className="brand"><span className="brand-mark">S</span><span>SonicMap</span></NavLink>
      <button className="menu-button" onClick={() => setOpen(!open)} aria-label="Toggle navigation">{open ? <X /> : <Menu />}</button>
      <nav className={open ? 'nav open' : 'nav'}>
        {links.map(([to, label, Icon]) => <NavLink key={to} to={to} onClick={() => setOpen(false)} className={({isActive}) => isActive ? 'active' : ''} end={to === '/'}><Icon size={16}/>{label}</NavLink>)}
      </nav>
      <span className="status"><i/>Research build</span>
    </header>
    <main>
      <Suspense fallback={<Loading/>}><Routes>
          <Route path="/" element={<Overview/>}/>
          <Route path="/embeddings" element={<Embeddings/>}/>
          <Route path="/similarity" element={<Similarity/>}/>
          <Route path="/generalization" element={<Generalization/>}/>
          <Route path="/playlists" element={<Playlists/>}/>
        </Routes></Suspense>
    </main>
    <footer><span>SonicMap · CPU-trained on real audio</span><span>Genre labels are a weak similarity proxy · No audio is hosted</span></footer>
  </div>
}
