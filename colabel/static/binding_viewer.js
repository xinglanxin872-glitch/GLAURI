/* Local-only monomer cartoon with binding-mode highlights. See BINDING_MODES.md. */
class BindingScene {
 constructor(){
  this.result=null;this.structure=null;this.request=0;this.left=null;
  this.set=document.querySelector('#view-set');this.chain=document.querySelector('#view-chain');
  this.type=document.querySelector('#view-type');this.mode=document.querySelector('#view-mode');
  this.status=document.querySelector('#view-status');
  this.capabilities={structure_available:false};
  this.capabilitiesReady=fetch('/api/capabilities').then(response=>{if(!response.ok)throw Error('Capability check failed');return response.json();}).then(data=>{this.capabilities=data;this.applyCapabilities();}).catch(()=>{
   document.querySelector('#data-mode').textContent='Reference-sequence view available. Coordinate availability could not be checked; restart the matching server to enable cartoons.';this.applyCapabilities(false);
  });
  this.set.onchange=()=>this.populateGroup();
  this.type.onchange=()=>this.populateInterface();
  this.chain.onchange=()=>this.loadStructure();
  this.mode.onchange=()=>this.draw();
  document.getElementById('residue-rule').onchange=()=>this.draw();
  document.getElementById('view-reset').onclick=()=>{if(this.left){this.left.zoomTo();this.left.render();}};
  document.getElementById('view-focus').onclick=()=>{const mode=this.currentMode();if(!mode||!this.structure)return;const mapped=this.mappedPositions(mode);if(mapped.length){this.left.zoomTo({resi:mapped});this.left.render();}};
  document.getElementById('view-pymol').onclick=()=>this.exportPyMOL();
 }
 options(select,items,value){select.replaceChildren();for(const [key,label] of items){const option=document.createElement('option');option.value=String(key);option.textContent=label;select.append(option);}if(items.some(([key])=>String(key)===String(value)))select.value=String(value);}
 applyCapabilities(updateText=true){
  const available=this.capabilities.structure_available;
  document.querySelector('#monomer-control').hidden=!available;
  document.querySelector('#cartoon-panel').hidden=!available;
  for(const id of ['view-focus','view-reset','view-pymol'])document.getElementById(id).hidden=!available;
  if(updateText)document.querySelector('#data-mode').textContent=available?
   'Index + H5: reference-sequence contact maps and monomer cartoons are available.':
   'Index-only view: contact maps, medoid chains, tables and JSON work without H5. To add cartoons, place the native file at data/bio_251210.h5 or restart with --h5 /path/to/bio_251210.h5.'+
   (this.capabilities.h5_available?' Install the structure dependencies: '+this.capabilities.missing_structure_dependencies.join(', ')+'.':'');
 }
 structureButtons(enabled){for(const id of ['view-focus','view-reset','view-pymol'])document.getElementById(id).disabled=!enabled;}
 setResult(result){
  this.result=result;document.querySelector('#visualization').hidden=!result.results.length;
  this.options(this.set,result.results.map((group,i)=>[i,group.structure_set]),0);
  if(result.results.length)this.populateGroup();else{this.request++;this.structure=null;}
 }
 group(){return this.result.results[Number(this.set.value)];}
 info(){return this.group().interfaces[this.type.value];}
 currentMode(){return this.info().modes.find(mode=>String(mode.pocket_id)===this.mode.value);}
 populateGroup(){
  const group=this.group(),entries=Object.entries(group.interfaces);
  const initial=entries.find(([,info])=>info.retained_modes)||entries.find(([,info])=>info.total_modes)||entries[0];
  this.options(this.type,entries.map(([type,info])=>[type,`${bindingNames[type]} (${info.retained_modes} modes)`]),initial[0]);
  this.populateInterface(true);
 }
 populateInterface(newGroup=false){
  const group=this.group(),info=this.info(),old=newGroup?null:this.chain.value;
  const preferred=[...group.matched_structure_keys,info.reference_key,...group.members].filter(Boolean);
  this.options(this.chain,[...new Set(preferred)].map(key=>[key,key+(key===info.reference_key?' · reference':'')]),old||group.matched_structure_keys[0]);
  this.options(this.mode,info.modes.map(mode=>[mode.pocket_id,`Mode ${mode.mode_id} · ${mode.residue_count} residues · support ${mode.support.toFixed(3)}`]),info.modes[0]?.pocket_id);
  this.loadStructure();
 }
 selectMode(groupIndex,type,pid){
  this.set.value=String(groupIndex);this.populateGroup();
  this.type.value=type;this.populateInterface();this.mode.value=String(pid);
  document.querySelector('#visualization').scrollIntoView({behavior:'smooth',block:'start'});
 }
 async loadStructure(){
  const ticket=++this.request;this.structure=null;this.structureButtons(false);this.drawReferenceTracks();
  await this.capabilitiesReady;
  if(ticket!==this.request)return;
  this.drawReferenceTracks();
  if(!this.capabilities.structure_available){this.clear();this.status.textContent='Reference maps use indexed labels only. H5 coordinates are optional and are needed only for the monomer cartoon and PyMOL export.';return;}
  if(!this.info().reference_key){this.clear();this.status.textContent='No contact records or reference labels for this interface. Select another interface.';return;}
  this.status.textContent='Loading monomer coordinates and mapping the reference sequence…';
  try{
   if(!window.$3Dmol)throw Error('The local 3D viewer could not load. Please refresh the page.');
   const query=new URLSearchParams({set:this.group().structure_set,type:this.type.value,key:this.chain.value});
   const response=await fetch('/api/structure?'+query);const data=await response.json();
   if(ticket!==this.request)return;if(!response.ok)throw Error(data.error);
   this.structure=data;this.structureButtons(true);
   if(!this.left){
    this.left=$3Dmol.createViewer(document.querySelector('#monomer-view'),{backgroundColor:'#f8fbfd',antialias:true});
    new ResizeObserver(()=>{this.left.resize();this.left.render();}).observe(document.querySelector('#viewer-grid'));
   }
   this.left.clear();this.left.addModel(data.pdb,'pdb');this.left.zoomTo();this.draw();
  }catch(error){if(ticket===this.request){this.structure=null;this.structureButtons(false);this.clear();this.status.textContent='The reference map remains available. Structure display failed: '+error.message;}}
 }
 clear(){if(this.left){this.left.clear();this.left.render();}}
 positions(mode){return document.querySelector('#residue-rule').value==='consensus'?mode.consensus_positions_1based:mode.reference_positions_1based;}
 drawReferenceTracks(){
  const info=this.info(),sequence=info.reference_sequence||'',length=sequence.length;
  const container=document.querySelector('#reference-tracks'),axis=document.querySelector('#reference-axis');container.replaceChildren();axis.replaceChildren();
  document.querySelector('#reference-fasta').textContent=sequence?'>'+info.reference_key+'\n'+sequence.match(/.{1,60}/g).join('\n'):'No reference sequence for this interface.';
  document.querySelector('#reference-description').textContent=length?
   `Reference ${info.reference_key} · FASTA length L = ${length} residues · ${info.retained_modes} / ${info.total_modes} modes pass the current filters. Display: ${document.querySelector('#residue-rule').selectedOptions[0].textContent}.`:
   'No reference sequence or contact records for this interface.';
  document.querySelector('#reference-hover').textContent='Hover over a strip to inspect its reference residue.';
  if(!length)return;
  const ns='http://www.w3.org/2000/svg';
  const svgNode=(tag,attributes,parent)=>{const item=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attributes))item.setAttribute(key,String(value));if(parent)parent.append(item);return item;};
  const scale=svgNode('svg',{viewBox:'0 0 1000 25',width:'100%',height:25,'aria-label':`Reference FASTA positions 1 to ${length}`},axis);
  const ticks=[...new Set([1,...[.25,.5,.75].map(f=>Math.round(1+(length-1)*f)),length])];
  for(const position of ticks){const x=length===1?0:1000*(position-1)/(length-1);const text=svgNode('text',{x,y:16,fill:'#526477','font-size':12,'text-anchor':position===1?'start':position===length?'end':'middle'},scale);text.textContent=String(position);}
  const selected=this.currentMode();
  for(const mode of info.modes){
   const active=mode.pocket_id===selected?.pocket_id,positions=this.positions(mode),sites=new Set(positions);
   const row=document.createElement('div');row.className='track-row'+(active?' selected-track':'');row.dataset.modeId=mode.mode_id;
   const button=document.createElement('button');button.type='button';button.className='track-label';button.textContent=`Mode ${mode.mode_id} · support ${mode.support.toFixed(3)}`;button.setAttribute('aria-pressed',String(active));
   const choose=()=>{this.mode.value=String(mode.pocket_id);this.draw();};button.onclick=choose;row.append(button);
   const track=svgNode('svg',{viewBox:`0 0 ${length} 20`,width:'100%',height:24,preserveAspectRatio:'none','data-reference-length':length,'data-contact-count':positions.length,'aria-label':`Mode ${mode.mode_id}, ${positions.length} contact sites over ${length} reference residues`,role:'img'});
   svgNode('rect',{x:0,y:3,width:length,height:14,fill:'#e4eaf0'},track);
   const runs=[];for(const position of positions){if(!runs.length||position>runs.at(-1)[1]+1)runs.push([position,position]);else runs.at(-1)[1]=position;}
   for(const [start,end] of runs){const segment=svgNode('rect',{x:start-1,y:3,width:end-start+1,height:14,fill:active?'#f59a23':'#427eaa','data-start':start,'data-end':end},track);svgNode('title',{},segment).textContent=`Reference ${start}–${end}: ${sequence.slice(start-1,end)}`;}
   track.onclick=choose;track.addEventListener('mousemove',event=>{const bounds=track.getBoundingClientRect(),position=Math.max(1,Math.min(length,Math.floor((event.clientX-bounds.left)/bounds.width*length)+1));document.querySelector('#reference-hover').textContent=`Mode ${mode.mode_id} · Reference ${sequence[position-1]}${position} / ${length} · ${sites.has(position)?'Contact':'Background'}`;});
   row.append(track);container.append(row);
  }
  if(!info.modes.length){const message=document.createElement('p');message.textContent='No modes pass the current filters. Reduce the residue or support threshold to inspect other modes.';container.append(message);}
 }
 mappedPositions(mode){const mapping=new Map(this.structure.reference_to_structure);return this.positions(mode).map(pos=>mapping.get(pos)).filter(pos=>pos!==undefined);}
 exportPyMOL(){
  if(!this.structure){this.status.textContent='Load a monomer structure first.';return;}
  const mode=this.currentMode(),mapped=mode?this.mappedPositions(mode):[];
  const lines=['# Standalone PyMOL cartoon; coordinates from the selected H5 monomer.',
   '# Residue IDs are sequential structure-sequence positions, not original PDB numbers.',
   'python','from pymol import cmd',`cmd.read_pdbstr(${JSON.stringify(this.structure.pdb)}, "binding_monomer")`,'python end',
   'hide everything, binding_monomer','dss binding_monomer','show cartoon, binding_monomer',
   'color gray80, binding_monomer','set cartoon_fancy_helices, 1','set cartoon_fancy_sheets, 1',
   'set cartoon_smooth_loops, 1','set cartoon_loop_radius, 0.2','bg_color white'];
  if(mapped.length)lines.push(`select binding_pocket, binding_monomer and chain A and resi ${mapped.join('+')}`,'color orange, binding_pocket');
  lines.push('orient binding_monomer','zoom binding_monomer');
  const anchor=document.createElement('a'),url=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/plain'}));
  anchor.href=url;anchor.download=this.structure.structure_key.replaceAll('/','_').replaceAll(':','_')+(mode?`_mode${mode.mode_id}`:'')+'_cartoon.pml';
  anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }
 draw(){
  this.drawReferenceTracks();
  if(!this.structure||!this.left)return;
  const mode=this.currentMode(),left=this.left;
  left.removeAllLabels();left.setHoverable({},false);left.setStyle({},{});
  const mappedMode=mode?new Set(this.mappedPositions(mode)):new Set();
  left.setStyle({},{cartoon:{style:'oval',arrows:true,tubes:false,ribbon:false,thickness:0.35,
   colorfunc:atom=>mappedMode.has(atom.resi)?'#f59a23':'#94a9bb'}});
  const data=this.structure;let description=`${data.structure_key} · ${data.residues.length} structure residues · ${data.atom_count} atoms; reference ${data.reference_key}; alignment identity ${(100*data.aligned_identity).toFixed(1)}%.`;
  if(mode){
   const positions=this.positions(mode),mapped=this.mappedPositions(mode),selection={resi:mapped};
   if(mapped.length){
    description+=` Mode ${mode.mode_id}: ${positions.length} reference sites; ${mapped.length} mapped, ${positions.length-mapped.length} unmapped. Mapped residues are highlighted in orange.`;
    left.setHoverable(selection,true,(atom,v)=>{
     const residue=data.residues[atom.resi-1];
     atom.hoverLabel=v.addLabel(`${residue.resname} · structure ${atom.resi} · reference ${residue.reference_position??'unmapped'}`,{position:atom,fontSize:12,backgroundColor:'#17384d',backgroundOpacity:0.9,fontColor:'white'});
    },(atom,v)=>{if(atom.hoverLabel){v.removeLabel(atom.hoverLabel);delete atom.hoverLabel;}});
   }else description+=` None of the ${positions.length} reference sites in mode ${mode.mode_id} can be mapped to this monomer.`;
  }else description+=' No modes pass the current filters; the monomer remains available.';
  this.status.textContent=description;left.render();
 }
}
window.BindingScene=BindingScene;
