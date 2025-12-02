# Zen Live

Real-time speech translation for broadcast news monitoring.

## Project Context

This is part of the [Zen AI model family](https://github.com/zenlm/zen), providing a hosted translation service via [Hanzo Node](https://github.com/hanzoai/hanzo-node) infrastructure.

## Architecture

```
Control Room Browser
        │
        │ WebRTC (audio/video)
        ▼
    Zen Live (FastRTC/FastAPI)
        │
        ├──► Hanzo Node API (recommended)
        │         │
        │         ▼
        │    Qwen3 LiveTranslate
        │
        └──► Direct DashScope API (fallback)
                  │
                  ▼
             Qwen3 LiveTranslate
```

## Backend Options

1. **Hanzo Node** (recommended): `HANZO_NODE_URL=http://host:3690`
2. **DashScope Direct**: `API_KEY=xxx`
3. **Zen Omni Local** (future): `ZEN_OMNI_PATH=/path/to/model`

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main FastAPI/FastRTC application |
| `index.html` | Control room web portal |
| `monitor.html` | Simplified broadcast display |
| `requirements.txt` | Python dependencies |
| `README.md` | User documentation |
| `LLM.md` | AI assistant context (this file) |

## Key Endpoints

- `/` - Control room UI
- `/monitor` - Broadcast monitor view
- `/api/status` - Health check
- `/broadcast/info` - Engineer integration guide
- `/audio/stream/{id}` - PCM16 audio stream
- `/outputs?webrtc_id={id}` - SSE transcripts

## Default Configuration

- **Source**: Spanish (news monitoring use case)
- **Target**: English
- **Voice**: Cherry
- **Audio**: PCM16, 24kHz, mono
- **Latency**: ~200-500ms

## Integration with Hanzo Node

When `HANZO_NODE_URL` is set, Zen Live can:
1. Query configured LLM providers from hanzo-node
2. Use node's API key management
3. Leverage node's monitoring/logging

Future: Native integration with zen-omni model for offline operation.

## Development Notes

- WebRTC via FastRTC library (Gradio ecosystem)
- CORS enabled for cross-origin control room access
- Audio subscribers pattern for broadcast streaming
- SSE for real-time transcript delivery

## UI/UX Analysis & Progressive Disclosure Implementation

### Current Layout Structure

**Header** (60px height, dark panel)
- Left: Zen Live logo with SVG icon
- Center: Navigation links (API Docs, Spec, GitHub)
- Right: Status bar (connection status, audio indicator, latency)

**Main Container** (2-column grid: 1fr 400px)
- **Left**: Video section with overlay captions
- **Right**: 400px sidebar with control panel (scrollable when content exceeds viewport)

### Detailed Control Panel Analysis

The sidebar contains 6 major functional sections (top to bottom):

1. **Control Panel Header** (Essential controls)
   - Source Language dropdown
   - Target Language dropdown
   - Voice selection
   - Input Source selector
   - Microphone selection
   - Audio Output selector
   - SRT Input URL (conditional visibility)
   - Start/Stop buttons

2. **Broadcast Output Settings** (Advanced - collapsible)
   - SRT Output URL
   - RTMP Output URL
   - NDI Output Name
   - Auto-reconnect checkbox
   - Transcript overlay checkbox
   - Save/Clear buttons
   - Info text (localStorage notice)

3. **Transcript Section** (Essential - informational)
   - Scrollable transcript log
   - Time-stamped entries with translations

4. **Audio Monitor** (Essential - informational)
   - Translation latency (current)
   - Average latency (last 10 frames)

5. **Audio Mixer** (Advanced - power user feature)
   - Mode toggle buttons (Source/Mix/Translated)
   - Source volume slider + mute button
   - Translated volume slider + mute button

6. **Output Endpoints** (Advanced - integration feature)
   - WebRTC Audio endpoint with copy button
   - Audio Stream (SSE) endpoint with copy button

### UI/UX Issues Identified

#### 1. **Visual Hierarchy & Clutter**
- **Issue**: The sidebar feels dense with 6 distinct sections in a 400px width
- **Severity**: Medium
- **Evidence**: 
  - Minimal spacing between sections (only 1px borders)
  - No visual separation besides borders
  - All text is same hierarchy level despite different importance
- **Impact**: Users scanning quickly may miss essential controls

#### 2. **Hidden Advanced Features**
- **Issue**: Broadcast Output Settings is collapsed (good), but Audio Mixer & Output Endpoints are always visible
- **Severity**: Medium-High
- **Evidence**:
  - Audio Mixer requires detailed understanding of audio routing
  - Output Endpoints are for integrations/broadcasting, not basic operation
  - These account for 25% of sidebar space
- **Impact**: Beginners overwhelmed with options not needed for basic streaming

#### 3. **Spacing & Alignment Issues**
- **Issue**: Inconsistent padding and spacing
- **Severity**: Low-Medium
- **Evidence**:
  - Control groups have 16px margin-bottom
  - Labels have 6px margin-bottom (inconsistent ratio)
  - Audio Mixer channel items have 10px margin-bottom
  - No consistent vertical rhythm
- **Impact**: Layout feels slightly off-balance; hard to establish visual pattern

#### 4. **Button Layout**
- **Issue**: Start/Stop buttons need better visual distinction
- **Severity**: Low
- **Evidence**:
  - Both buttons are flex: 1 (equal width)
  - Stop button appears after Start in tab order
  - No disabled state styling is obvious
- **Impact**: Could accidentally click Stop during setup

#### 5. **Responsive Behavior**
- **Issue**: On tablets/1024px breakpoints, layout becomes unwieldy
- **Severity**: Medium
- **Evidence**:
  - Media query at @media (max-width: 1024px) switches to single column
  - Sidebar becomes 50% of viewport height
  - No accommodation for smaller sidebars or collapsing sections
- **Impact**: Unusable on iPad-sized screens with current design

#### 6. **Transcript Log Scrolling**
- **Issue**: Transcript section flex: 1 takes remaining space
- **Severity**: Low
- **Evidence**:
  - If many translations accumulate, section expands
  - May push Audio Mixer/Endpoints off screen
  - Unclear at what point user needs to scroll sidebar
- **Impact**: On smaller viewports, bottom sections become inaccessible

#### 7. **Font Size & Readability**
- **Issue**: Labels are very small (0.8rem uppercase)
- **Severity**: Low
- **Evidence**:
  - Control group labels: 0.8rem, uppercase, 0.5px letter-spacing
  - Sidebar header: 0.85rem, uppercase
  - Help text: 0.75rem
- **Impact**: Accessibility concern; hard to read for vision-impaired users

### Spacing & Alignment Details

Current Vertical Spacing:
```
Control Groups:
  - Label: 0.8rem, margin-bottom: 6px
  - Input/Select: padding: 10px 12px
  - Help text: 0.75rem, margin-top: 4px
  - Group margin-bottom: 16px

Sections:
  - Sidebar header: 16px padding, border-bottom: 1px
  - Controls section: 16px padding
  - Between sections: 1px borders, no margin

Audio Mixer:
  - Channel margin-bottom: 10px (inconsistent with 16px standard)
  - Channel label: 80px fixed width
  - Slider: flex: 1
  - Value display: 40px fixed width
  - Mute button: 4px 8px padding (too small)
```

### Control Categorization

**Essential (Always Visible)**
- Source/Target Language
- Voice Selection
- Input Source Selection
- Start/Stop Buttons
- Transcript Log
- Connection Status (header)

**Important (Default Visible)**
- Microphone Selection
- Audio Output Selection
- Latency Displays
- Translation Status

**Advanced (Should be Collapsed/Hidden)**
- Audio Mixer (requires audio engineering knowledge)
- Output Endpoints (integration/broadcast feature)
- Broadcast Output Settings (for streaming engineers)
- SRT Input URL (conditional, only when SRT selected)

**Debug/Info (Could be toggled)**
- Endpoint URLs (copy functionality)
- Average latency vs single latency
- Audio mode selection

### Recommended Progressive Disclosure Implementation

**Simple Mode (Default)**
- Header with status
- Video section (full)
- Sidebar with:
  - Language controls (3 dropdowns: source, target, voice)
  - Input controls (2 dropdowns: source type, device)
  - Start/Stop buttons
  - Transcript log
  - Status display (latency, connection)
  
**Advanced Mode (Toggle)**
- All simple mode items +
- Audio Mixer section (with icon indicating "advanced")
- Broadcast Output Settings (expanded by default in advanced mode)
- Output Endpoints section

**Button/Toggle Placement**
- Add "Advanced" toggle button in header (top-right, before Settings)
- Or: Right-click context menu to toggle
- Or: Keyboard shortcut (Ctrl+Shift+A)
- Store preference in localStorage

### CSS Improvements Needed

1. **Vertical Rhythm**: Establish consistent spacing unit (8px or 4px)
2. **Section Separation**: Add 8-12px margin-top to major sections, not just 1px borders
3. **Button States**: Make disabled state more obvious with reduced opacity + cursor: not-allowed
4. **Touch Targets**: Mute buttons should be at least 36x36px (currently ~20px)
5. **Label Contrast**: Increase sidebar header font-size from 0.85rem to 1rem
6. **Focus States**: Add :focus-visible styling for keyboard navigation

### Accessibility Issues

1. **Color Contrast**: Verify text-muted (#666) on bg-panel (#111) meets WCAG AA
   - Current ratio appears too low for body text
   - Help text (#666 on #0a0a0a) is even worse

2. **Missing ARIA Labels**: No aria-label attributes on:
   - Fullscreen toggle button
   - Mute buttons (just "M" text)
   - Copy endpoint buttons
   - Mixer mode toggle buttons

3. **Keyboard Navigation**: Tab order may skip elements in collapsible sections

### Specific File Recommendations

**index.html Changes Needed:**
- Line 822-843: Audio Mixer section → wrap in `<details>` or conditional div
- Line 845-861: Output Endpoints → wrap in conditional div or `<details>`
- Line 677-802: Broadcast Settings → already collapsible, but could be hidden in simple mode
- Add toggle button to header (around line 635-639)
- Consider grouping related controls with `<fieldset>` elements

**CSS Changes:**
- Add classes for `.simple-mode` and `.advanced-mode`
- Implement conditional visibility:
  ```css
  body.simple-mode .audio-mixer { display: none; }
  body.simple-mode .output-endpoints { display: none; }
  body.advanced-mode .broadcast-settings { margin-top: 0; }
  ```
- Improve button styling with better visual feedback
- Add focus ring styling for accessibility

## Links

- GitHub: https://github.com/zenlm/zen-live
- Hanzo Node: https://github.com/hanzoai/hanzo-node
- Zen Models: https://github.com/zenlm/zen
- Hanzo AI: https://hanzo.ai
