"use client";

import { useEffect, useState, useRef } from "react";

interface SmokeEvent {
  Frame: number;
  Timestamp: string;
  "Video Seconds": number;
  "Smoke Type": string;
  "Master ID": string;
  "Vehicle Type": string;
  "License Plate": string;
  "Cam ID": number | string;
  Source: string;
  CropBase64: string;
  FullFrameBase64?: string;
}

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [frames, setFrames] = useState<Record<string, string>>({});
  const [events, setEvents] = useState<SmokeEvent[]>([]);
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connectWs = () => {
      const ws = new WebSocket("ws://localhost:8000/ws/stream");

      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.frames) {
            setFrames((prev) => ({ ...prev, ...data.frames }));
          }
          
          if (data.events && Array.isArray(data.events)) {
            setEvents((prev) => {
              const newList = [...data.events, ...prev];
              return newList.slice(0, 50);
            });
          }
        } catch (e) {
          console.error("Error parsing WS message", e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connectWs, 3000);
      };

      wsRef.current = ws;
    };

    connectWs();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Determine grid columns based on number of cameras
  const numCameras = Math.max(1, Object.keys(frames).length);
  const gridCols = numCameras > 1 ? 2 : 1;

  return (
    <main className="dashboard-container">
      <header className="dashboard-header">
        <h1>Motorway CCTV Operations</h1>
        <div className="status-badge">
          <div className={`status-indicator ${isConnected ? "connected" : ""}`}></div>
          {isConnected ? "SYSTEM ONLINE" : "CONNECTING..."}
        </div>
      </header>

      <div className="dashboard-grid">
        {/* Left Column - Live Streams Grid */}
        <section className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 className="panel-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 7l-7 5 7 5V7z"></path>
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            </svg>
            Live Feeds ({numCameras})
          </h2>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: `repeat(${gridCols}, 1fr)`, 
            gap: '1rem',
            width: '100%',
            flex: 1
          }}>
            {Object.keys(frames).length === 0 ? (
              <div className="video-container" style={{ gridColumn: "1 / -1" }}>
                <div className="video-placeholder">
                  <div className="spinner"></div>
                  Waiting for video streams...
                </div>
              </div>
            ) : (
              Object.entries(frames).map(([camId, frameData]) => (
                <div key={camId} className="video-container">
                  <img src={`data:image/jpeg;base64,${frameData}`} alt={`Cam ${camId}`} className="video-feed" />
                  <div style={{
                    position: 'absolute', top: 10, left: 10, 
                    background: 'rgba(0,0,0,0.7)', padding: '4px 8px', 
                    borderRadius: 4, fontSize: '12px', fontWeight: 'bold'
                  }}>
                    CAM {camId}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Right Column - Events */}
        <section className="glass-panel">
          <h2 className="panel-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            Real-Time Alerts
          </h2>
          
          <div className="events-list">
            {events.length === 0 ? (
              <div style={{color: "var(--text-secondary)", textAlign: "center", padding: "2rem"}}>
                No active events detected.
              </div>
            ) : (
              events.map((ev, i) => {
                const isSevere = ev["Smoke Type"]?.toLowerCase().includes("black") || ev["Smoke Type"]?.toLowerCase().includes("blue");
                
                return (
                  <div key={i} className={`event-card ${isSevere ? "severe" : ""}`}>
                    <div className="event-header">
                      <span>{ev.Timestamp || `Frame ${ev.Frame}`}</span>
                      <span>Cam {ev["Cam ID"]}</span>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', margin: '0.5rem 0' }}>
                      {ev.FullFrameBase64 ? (
                         <div style={{ position: 'relative', cursor: 'zoom-in' }} onClick={() => setSelectedFrame(`data:image/jpeg;base64,${ev.FullFrameBase64}`)}>
                           <img 
                             src={`data:image/jpeg;base64,${ev.FullFrameBase64}`} 
                             alt="Full Frame Evidence" 
                             style={{ width: '120px', height: '80px', objectFit: 'cover', borderRadius: '4px', border: '1px solid var(--glass-border)' }}
                           />
                           <div style={{ position: 'absolute', bottom: '4px', right: '4px', background: 'rgba(0,0,0,0.6)', color: 'white', fontSize: '9px', padding: '2px 4px', borderRadius: '2px', pointerEvents: 'none' }}>
                             Click to Enlarge
                           </div>
                         </div>
                      ) : ev.CropBase64 ? (
                         <img 
                           src={`data:image/jpeg;base64,${ev.CropBase64}`} 
                           alt="Vehicle Crop" 
                           style={{ width: '80px', height: '80px', objectFit: 'cover', borderRadius: '4px', border: '1px solid var(--glass-border)' }}
                         />
                      ) : (
                         <div style={{ width: '80px', height: '80px', background: 'rgba(0,0,0,0.3)', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px', textAlign: 'center', color: '#666' }}>No Image</div>
                      )}
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        <div className="event-plate">
                          {ev["License Plate"] || "UNREADABLE"}
                        </div>
                        <div className="event-details">
                          <span className={`tag ${isSevere ? 'red' : 'orange'}`}>
                            {ev["Smoke Type"] || "Unknown Smoke"}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>
      </div>

      {selectedFrame && (
        <div className="lightbox-overlay" onClick={() => setSelectedFrame(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setSelectedFrame(null)}>×</button>
            <img src={selectedFrame} alt="Full Frame Evidence" className="lightbox-image" />
          </div>
        </div>
      )}
    </main>
  );
}
