(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[931],{1591:function(e,t,n){Promise.resolve().then(n.bind(n,2443))},2443:function(e,t,n){"use strict";n.r(t),n.d(t,{default:function(){return j}});var i=n(7437),s=n(2265),l=n(6648),r=n(920);let a=s.createContext([{id:(0,r.Z)(),createdAt:new Date,updatedAt:new Date,submittedAt:new Date,ImageAnnotations:new Map,ImageSelection:new Map},()=>{}]);var d=e=>{let{imageUrl:t,prompt:n}=e,[r,d]=(0,s.useContext)(a),c=(0,s.useRef)(null),[o,h]=(0,s.useState)("Add"),[A,g]=(0,s.useState)({tool:"Add",points:[]}),[u,m]=(0,s.useState)([]),[x,f]=(0,s.useState)(0),b=()=>{A.points.length<=2||(f(x+1),g({tool:o,points:[]}))};return(0,s.useEffect)(()=>{console.log("first usesEffect");let e=c.current;if(!e)return;let n=e.getContext("2d");if(!n)return;n.clearRect(0,0,e.width,e.height),e.style.backgroundColor="transparent",m([]),f(0),g({tool:"Add",points:[]});let i=r.ImageAnnotations.get(t.src);console.log(t.src),i&&(h(i[0].tool),m(i),g(i[0]))},[r.ImageAnnotations,t.src]),(0,s.useEffect)(()=>{console.log("second useEffect");let e=c.current;if(!e)return;let t=e.getContext("2d");if(t)for(let n of(t.clearRect(0,0,e.width,e.height),u)){if(t.beginPath(),0===n.points.length)return;switch(t.moveTo(n.points[0].x,n.points[0].y),n.points.forEach(e=>{t.lineTo(e.x,e.y),t.arc(e.x,e.y,3,0,2*Math.PI)}),t.closePath(),n.tool){case"Add":t.strokeStyle="rgba(0, 0, 255, 1)",t.fillStyle="rgba(0, 0, 255, 0.1)";break;case"Remove":t.strokeStyle="rgba(255, 0, 0, 1)",t.fillStyle="rgba(255, 0, 0, 0.1)";break;case"Change":t.strokeStyle="rgba(0, 255, 0, 1)",t.fillStyle="rgba(0, 255, 0, 0.1)"}t.stroke(),t.fill()}},[u,A,t.src,o]),(0,i.jsxs)("div",{className:"flex-row p-4 rounded-lg dark:text-black  items-center justify-center",children:[(0,i.jsxs)("h1",{className:"text-xl mb-16 text-wrap max-w-2xl",children:["Markieren sie nun bitte die Bereiche im Gesicht der Person, die sich, Ihrer Meinung nach, \xe4ndern m\xfcssen, damit die Person als “",(0,i.jsx)("span",{className:"font-bold",children:"l\xe4chelnd"}),"” klassifiziert werden sollte. Daf\xfcr k\xf6nnen sie einfach durch Klicken beliebig viele Bereiche festlegen."]}),(0,i.jsx)("div",{className:"rounded flex items-center justify-center",children:(0,i.jsxs)("div",{className:"relative w-[".concat(512,"px] h-[").concat(512,"px] cursor-crosshair border-2  rounded"),children:[(0,i.jsx)("canvas",{ref:c,width:512,height:512,onClick:e=>{let n=e.target.getBoundingClientRect(),i=e.clientX-n.left,s=e.clientY-n.top,l={tool:A.tool,points:[...A.points,{x:i,y:s}]};u[x]=l,r.ImageAnnotations.set(t.src,u),g(l)},className:"absolute top-0 left-0"}),(0,i.jsx)(l.default,{className:" rounded",src:t,alt:"Image to be anntoated",width:512,height:512}),(0,i.jsxs)("div",{className:"absolute right-0 top-1/2 transform -translate-y-1/2 space-y-2 p-1 bg-white shadow-lg rounded-l-md cursor-pointer",children:[(0,i.jsx)("div",{className:"bg-blue-200 hover:bg-blue-300 rounded-full p-2",children:(0,i.jsx)(l.default,{onClick:()=>{g({tool:o,points:[]}),m([]),f(0),r.ImageAnnotations.delete(t.src)},src:"/undo.svg",alt:"Undo",width:20,height:20})}),(0,i.jsx)("div",{className:"bg-blue-200 hover:bg-blue-300 rounded-full p-2",children:(0,i.jsx)(l.default,{onClick:()=>b(),src:"/enter.svg",alt:"Undo",width:20,height:20,title:"Markiere Bereiche die relevant sind"})})]})]})})]})},c=e=>{let{steps:t}=e,[n,l]=(0,s.useState)(0),r=(n+1)/t.length*100;return(0,i.jsxs)("div",{className:"w-full h-screen",children:[(0,i.jsx)("div",{className:"bg-goKiBG w-full rounded-lg flex flex-col items-center p-8 h-5/6",children:t[n]}),(0,i.jsx)("div",{className:"mt-2 flex justify-between",children:(0,i.jsxs)("div",{className:"mt-4 w-full h-2 bg-gray-200",children:[(0,i.jsx)("div",{className:"h-full bg-goKiAccent",style:{width:"".concat(r,"%")}})," "]})}),(0,i.jsxs)("div",{className:"mt-2 flex justify-between",children:[n>0?(0,i.jsx)("button",{className:"bg-goKiAccent hover:bg-[#965335] text-white font-bold py-2 px-4 rounded mr-2",onClick:()=>{l(e=>Math.max(e-1,0))},children:"Back"}):(0,i.jsx)("button",{className:"bg-gray-500 text-white font-bold py-2 px-4 rounded mr-2",disabled:!0,children:"Back"}),n<t.length-1?(0,i.jsx)("button",{className:"bg-goKiPrimary hover:bg-goKiPrimary text-white font-bold py-2 px-4 rounded",onClick:()=>{l(e=>Math.min(e+1,t.length-1))},children:"Next"}):(0,i.jsx)("button",{className:"bg-goKiPrimary text-white hover:bg-goKiPrimary font-bold py-2 px-4 rounded",onClick:()=>l(0),children:"Zur\xfcck zum Anfang"})]})]})},o=()=>(0,i.jsxs)("div",{className:"mt-10 max-w-2xl h-full",children:[(0,i.jsx)("div",{className:"flex mb-4 items-center",children:(0,i.jsx)("h1",{className:"text-2xl font-bold text-goKiPrimary ",children:"Kontrafaktische Erkl\xe4rungen"})}),(0,i.jsxs)("p",{className:"mb-4 text-xl",children:["Mit kontrafaktische Erkl\xe4rungen versucht man die Verhaltensweisen eines Modells verst\xe4ndlich zu machen."," "]}),(0,i.jsx)("p",{className:"mb-4 text-xl",children:"Dabei wird untersucht wie sich die Eingabe des Modells ver\xe4ndern muss, damit das Modell eine andere Entscheidung trifft. Dadurch kann man nachvollziehen auf welche Merkmale das Modell besonders acht gibt."}),(0,i.jsx)("p",{className:"mb-4 text-xl",children:"In dieser Anwendung erkl\xe4ren wir das Verhalten eines KI-Modells das darauf trainiert wurde zu entscheiden, ob eine Person l\xe4chelt oder nicht l\xe4chelt. Daf\xfcr wollen wir herausfinden welche Bereiche im Gesicht der Personen sich ver\xe4ndern m\xfcssen, damit das Modell seine Entscheidung \xe4ndert. Daf\xfcr Vergleichen wir die kontrafaktische Erkl\xe4rung einer XAI-Methode und Ihrer eigenen Vorstellung des Konzepts “L\xe4cheln”."})]}),h=e=>{let{imageId:t,numImages:n}=e,[r,a]=(0,s.useState)(0),[d,c]=(0,s.useState)({width:256,height:256}),o=((e,t)=>{let n=[];for(let i=0;i<t;i++)n.push("/sequences/".concat(e,"/").concat(i,".png"));return n})(t,n);return(0,s.useEffect)(()=>{let e=()=>{let e=Math.floor(window.innerWidth/n)-20;c({width:e,height:e})};return e(),window.addEventListener("resize",e),()=>{window.removeEventListener("resize",e)}},[n]),(0,i.jsxs)("div",{className:"flex p-8 rounded-lg  w-full dark:text-black items-center justify-center flex-col",children:[(0,i.jsxs)("h1",{className:"text-xl  mb-16 text-center ",children:["Diese Bilderreihe zeigt ganz links das Originalbild. Die Bilder rechts davon sind AI generierte Bilder, in denen das Merkmal “",(0,i.jsx)("span",{className:"font-bold",children:"L\xe4cheln"}),"” stufenweise intensiviert wurde. W\xe4hlen Sie, das von links aus, erste Bild in dem die Person, Ihrer Meinung nach, ",(0,i.jsx)("span",{className:"font-bold",children:"l\xe4chelt"}),"."]}),(0,i.jsx)("div",{className:" w-full flex space-x-1 justify-center",children:o.map((e,t)=>(0,i.jsx)(l.default,{src:e,alt:"Option ".concat(t+1),width:d.width,height:d.height,onClick:()=>{a(t)},className:"cursor-pointer ".concat(t===r?"border-4 border-goKiPrimary":"")},t))}),(0,i.jsx)("button",{onClick:()=>{a(-1)},className:"px-4 py-2 rounded mt-4 font-bold ".concat(-1===r?"bg-goKiPrimary border-4 border-goKiAccent text-white":"bg-white text-black opacity-70 hover:opacity-100 hover:ring-2 ring-gobg-goKiPrimary"),children:"None is smiling"})]})},A={src:"/_next/static/media/27300_1.0_misclassified.1ca47a08.png",height:256,width:256,blurDataURL:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAMAAADz0U65AAAAXVBMVEUmHBrCj3a3fmuXXy5FNSZnSzp4X0RTQTDTloCFWj0yIhxhSjs2MjWbfVNuU0C1jn1IOjK/hW6ugmakeGOUYjFsV0oMCAefaTAXDgtkRii6jHGqi3CkiXzGkoaRZVQysEUVAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAQ0lEQVR4nAXBhQHAMAwDMBfSpExj+v/MSbDcLGezgItz9twNKN+Xj97DjadWjiumfpUqSPi0SKcNSFNEBwBtqE5HwA9nQgLJrzo+gwAAAABJRU5ErkJggg==",blurWidth:8,blurHeight:8},g=n(5678),u={src:"/_next/static/media/0.90b0dea0.png",height:256,width:256,blurDataURL:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAMAAADz0U65AAAAYFBMVEUnHht4X0S7gGskFxTTloCFWjxmSjpSQDBFNifCjXSXXy02KCRgSjtuU0Cvi3SdeGCugmY/OTmUYjBsVkkLBwafaS+df0oXDQpkRie6jHG1j4KzfG2kiXzGkoame2GRZVTBD+RzAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARUlEQVR4nAXBBwKAIAwEsAMKbRkC7u3/f2kCYRFe4wguxsi+RIQ6HD57D9Puc+A84bWPakHCZ4lsmB16JbqSA3pTDRvwA2ibAtDrq4eOAAAAAElFTkSuQmCC",blurWidth:8,blurHeight:8},m={src:"/_next/static/media/3.175fafee.png",height:256,width:256,blurDataURL:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAMAAADz0U65AAAAUVBMVEUvJh6OZEpAOTfQloO8inVcRjJLPCxnTz2ecTZDNCZuWD9+WkuzjX94YUmmhWq3h3G0fGyYd13CkH6ddWGGYjkUDgqghU04LSEbEg2fhXvFk4AcJhxkAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAQ0lEQVR4nBXGSRLAIAgEwFFQwF2z5/8PTaVPDYt2xVMVQVKyoAqZrTvZHWp7SglSMcdizvBwnV7OB+A3ohHxZ7Hchg9bJQJuLbh0sAAAAABJRU5ErkJggg==",blurWidth:8,blurHeight:8};function x(){let[e,t]=(0,s.useState)(!1);return(0,i.jsxs)("div",{className:"selection:h-full mt-10 dark:text-black",children:[(0,i.jsxs)("h1",{className:"text-xl mb-16 text-wrap max-w-2xl",children:["Sie k\xf6nnen nun vergleichen, ob Sie dieselbe Vorstellung vom Konzept „",(0,i.jsx)("span",{className:"font-bold",children:"L\xe4cheln"}),"“ haben wie das KI-Modell. Klicken Sie auf die roten Knopf, um die kontrafaktische Erkl\xe4rung der XAI-Methode zu erzeugen, die die Klassifizierung des Modells in „",(0,i.jsx)("span",{className:"font-bold",children:"L\xe4cheln"}),"“ \xe4ndert."]}),(0,i.jsxs)("div",{className:"grid grid-cols-3 my-auto space-x-2",children:[(0,i.jsx)(l.default,{src:u,alt:"Image Before"}),(0,i.jsx)("button",{className:"bg-goKiAccent border border-1 border-[#965335] text-white p-4 rounded-full",onClick:()=>{t(!0)},children:"Generate Counterfactual"}),e?(0,i.jsx)(g.E.div,{initial:{opacity:0,filter:"blur(20px)"},animate:{opacity:1,filter:"blur(0px)"},transition:{duration:1},children:(0,i.jsx)(l.default,{src:m,alt:"Image After"})}):(0,i.jsx)("div",{style:{width:"100%",height:"100%"}})]})]})}var f=()=>(0,i.jsxs)("div",{className:"p-5 rounded-lg mt-10 max-w-2xl h-full dark:text-black",children:[(0,i.jsxs)("p",{className:"mb-8 text-xl",children:["Das Modell hat die Person im Bild als ",(0,i.jsx)("span",{className:"font-bold",children:"nicht l\xe4chelnd"})," klassifiziert. Was muss sich, Ihrer Meinung nach, ver\xe4ndern, dass die Person als ",(0,i.jsx)("span",{className:"font-bold",children:"l\xe4chelnd"})," klassifiziert werden kann?"]}),(0,i.jsx)(l.default,{src:A,alt:"Person nicht l\xe4chelnd",width:300,height:300,className:"mx-auto mb-4"})]}),b={src:"/_next/static/media/27300_annot.9fa01c5b.png",height:480,width:640,blurDataURL:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAGCAMAAADJ2y/JAAAAPFBMVEX+///GwLywppzZ1tLWuaXt6+nq2dbf2NXRzcnw5+DDq5u1pJXz8/KWfHPjpIDTkXmzl4aUj4zdtJ2alZUEV95DAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuNSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/xnp5ZAAAANElEQVR4nBXGSRLAIAgAwVFAQY1b/v/XVPrUYO4SA0g591YBV+2tAHLnfv8kXc8JwMKkDj4degEqC1l0LgAAAABJRU5ErkJggg==",blurWidth:8,blurHeight:6},w={src:"/_next/static/media/27300_sel.cc8ad193.png",height:1014,width:1720,blurDataURL:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAMAAABPT11nAAAAS1BMVEXz9PTcu7ellIyzl4r8/f3t7Ozr6enu7u73+frd29n4xJjGu7K7sKXVy8TVy8zHQyW1qqC2RESPcFulhX/srqH8gC793r7SkpLbIwhw2n8fAAAACXBIWXMAABYlAAAWJQFJUiTwAAAAMUlEQVR4nAXBBwIAEAwEsDNKa2/+/1IJLBx55QAQcx5NQBSTqUaDRc/+doCVou4K5wMZ8gF4OKC9mQAAAABJRU5ErkJggg==",blurWidth:8,blurHeight:5};function p(){return(0,i.jsxs)("div",{className:"selection:h-full mt-10 dark:text-black inline-block",children:[(0,i.jsx)("h1",{className:"text-2xl mx-auto mb-16 text-wrap max-w-2xl text-center text-goKiPrimary",children:"Stimmen Sie mit der Erkl\xe4rung \xfcberein?"}),(0,i.jsx)("h1",{className:"text-xl mx-auto mb-16 text-wrap max-w-2xl text-center",children:"Hier sehen Sie wie Ihre eigene kontrafaktische Erkl\xe4rung im Vergleich zu anderen Teilnehmenden und der XAI Methode abschneidet."}),(0,i.jsxs)("div",{className:"grid grid-cols-2",children:[(0,i.jsxs)("div",{children:[(0,i.jsx)(l.default,{className:"mb-2",src:b,alt:"Image Before"}),(0,i.jsx)("p",{children:"Die meisten Teilnehmenden stimmen mit der XAI Methode \xfcberein (T\xfcrkis umrahmtes Bild). Die l\xe4chelnde Person, die die Mehrzahl ausw\xe4hlen, ist die gleiche, wie die kontrafaktische Erkl\xe4rung der XAI Methode."})]}),(0,i.jsxs)("div",{children:[(0,i.jsx)(l.default,{className:"mb-2",src:w,alt:"Image Before"}),(0,i.jsx)("p",{children:"Die rot markierten Stellen sind die, die die meisten Teilnehmenden f\xfcr ein L\xe4cheln ver\xe4ndern wollen."})]})]})]})}function j(){let e=[(0,i.jsx)(o,{},1),(0,i.jsx)(f,{},2),(0,i.jsx)(h,{numImages:6,imageId:"27300"},2),(0,i.jsx)(d,{imageUrl:A,prompt:""},3),(0,i.jsx)(x,{},2),(0,i.jsx)(p,{},2)];return(0,i.jsx)("main",{className:"flex min-h-screen text-goKiAccent flex-col items-center justify-between p-20 bg-goKiBG",children:(0,i.jsx)(c,{steps:e})})}}},function(e){e.O(0,[932,971,23,744],function(){return e(e.s=1591)}),_N_E=e.O()}]);