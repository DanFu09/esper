import React from 'react';
import {boundingRect} from './FrameView.jsx';
import ClipView from './ClipView.jsx';
import {observer, inject} from 'mobx-react';
import {toJS} from 'mobx';
import keyboardManager from 'utils/KeyboardManager.jsx';
import Consumer from 'utils/Consumer.jsx';
import {FrontendSettingsContext, BackendSettingsContext, SearchContext} from './contexts.jsx';
import Select from './Select.jsx';
import {PALETTE} from 'utils/Color.jsx';

@observer
class MarkerView extends React.Component {
  render() {
    return <Consumer contexts={[FrontendSettingsContext]}>{ frontendSettings => {
        let {t, w, h, mw, mh, mf, label, type, color, ..._} = this.props;
        let range = frontendSettings.get('timeline_range');
        let time_to_x = (t) => w/2 + (t - this.props.currentTime) / (range/2) * w/2;
        let x = time_to_x(t);
        let margin = mw;
        if (0 <= x && x <= w) {
          if (type == 'open') {
            let points = `${mw*2},${margin} 0,${margin} 0,${mh-2*margin} ${mw*2},${mh-2*margin}`;
            return (<g transform={`translate(${x}, 0)`}>
              <polyline fill="none" stroke={color} strokeWidth={mw} points={points} />
              <text x={mw+4} y={h/2} alignmentBaseline="middle" fontSize={mf}>{label}</text>
            </g>);
          } else {
            let points = `${-mw*2},${margin} 0,${margin} 0,${mh-2*margin} ${-mw*2},${mh-2*margin}`;
            return (<g transform={`translate(${x}, 0)`}>
              <polyline fill="none" stroke={color} strokeWidth={mw} points={points} />
              <text x={-(mw+4)} y={h/2} alignmentBaseline="middle" textAnchor="end" fontSize={mf}>{label}</text>
            </g>);
          }
        } else {
          return <g />;
        }
    }}</Consumer>
  }
}

@observer
class TrackView extends React.Component {
  _mouseX = -1
  _mouseY = -1

  _onKeyPress = (e) => {
    if (keyboardManager.locked()) {
      return;
    }

    let rect = boundingRect(this._g);
    let [x, y] = this._localCoords();
    if (!(0 <= x && x <= rect.width && 0 <= y && y <= rect.height)) {
      return;
    }

    let chr = String.fromCharCode(e.which);
    this.props.onKeyPress(chr, this.props.i);
  }

  _onMouseMove = (e) => {
    this._mouseX = e.clientX;
    this._mouseY = e.clientY;
  }

  _localCoords = (e) => {
    let rect = boundingRect(this._g);
    return [this._mouseX - rect.left, this._mouseY - rect.top];
  }

  componentDidMount() {
    document.addEventListener('keypress', this._onKeyPress);
    document.addEventListener('mousemove', this._onMouseMove);
  }

  componentWillUnmount() {
    document.removeEventListener('keypress', this._onKeyPress);
    document.removeEventListener('mousemove', this._onMouseMove);
  }

  render() {
    return <Consumer contexts={[FrontendSettingsContext, BackendSettingsContext, SearchContext]}>{(frontendSettings, backendSettings, searchResult) => {
        let {track, w, h, mw, mh, mf, video, ..._} = this.props;
        let start = track.min_frame / video.fps;
        let end = track.max_frame / video.fps;

        let range = frontendSettings.get('timeline_range');
        let time_to_x = (t) => w/2 + (t - this.props.currentTime) / (range/2) * w/2;
        let x1 = time_to_x(start);
        let x2 = time_to_x(end);

        let color;
        if (track.gender_id !== undefined) {
          color = searchResult.gender_colors[searchResult.genders[track.gender_id].name];
        } else if (track.identity !== undefined) {
          let ident_colors = PALETTE;
          color = ident_colors[track.identity % ident_colors.length];
        } else {
          color = '#eee';
        }

        let text = null;
        if (track.identity !== undefined) {
          text = track.identity;
        } else if (track.thing_id !== undefined) {
          text = backendSettings.things['topic'][track.thing_id];
        }

        return (
          <g ref={(n) => {this._g = n;}}>
            <rect x={x1} width={x2-x1} y={0} height={h} fill={color} />
            {range < 600
             ? <g>
               <line x1={x1} y1={0} x2={x1} y2={h} stroke="black" />
               <line x1={x2} y1={0} x2={x2} y2={h} stroke="black" />
               {text !== null
                ? <text x={x1+2} y={h/2} textAnchor="start" alignmentBaseline="middle">{text}</text>
                : <g />}
             </g>
             : <g />}
          </g>
        );
    }}</Consumer>;
  }
}

@observer
export default class TimelineView extends React.Component {
  state = {
    currentTime: 0,
    clickedTime: -1,
    displayTime: -1,
    startX: -1,
    startY: -1,
    trackStart: -1,
    moused: false,
    showSelect: false
  }

  _videoPlaying = false;
  _lastPlaybackSpeed = null;
  _undoStack = [];

  _onTimeUpdate = (t) => {
    if (t != this.state.currentTime) {
      this.setState({currentTime: t});
    }
  }

  _localCoords = (e) => {
    let rect = boundingRect(this._svg);
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  _onMouseDown = (e) => {
    let [x, y] = this._localCoords(e);
    this.setState({startX: x, startY: y, clickedTime: this.state.currentTime});
  }

  _onMouseMove = (e) => {
    if (this.state.startX != -1) {
      let [x, y] = this._localCoords(e);
      let dx = x - this.state.startX;
      let dt = this._frontendSettings.get('timeline_range') * dx / boundingRect(this._svg).width * -1;
      this.setState({
        displayTime: this.state.clickedTime + dt,
        currentTime: this.state.clickedTime + dt
      });
    }
  }

  _onMouseUp = (e) => {
    this.setState({startX: -1, startY: -1});
  }

  _timelineOnMouseOut = (e) => {
    if (this.state.startX != -1) {
      this.setState({startX: -1, startY: -1});
    }
  }

  _video = () => {
    return this._searchResult.videos[this.props.group.elements[0].video];
  }

  _onVideoPlay = () => {
    this._videoPlaying = true;
  }

  _onVideoStop = () => {
    this._videoPlaying = false;
  }

  _pushState = () => {
    this._undoStack.push(_.cloneDeep(toJS(this.props.group.elements)));

    // Keep the stack small to avoid using too much memory
    let MAX_UNDO_STACK_SIZE = 10;
    if (this._undoStack.length > MAX_UNDO_STACK_SIZE) {
      this._undoStack.shift();
    }
  }

  _onTrackKeyPress = (e, i) => {
    let chr = String.fromCharCode(e.which);

    // Change track gender
    if (chr == 'g') {
      this._pushState();
      let track = this.props.group.elements[i];
      let keys = _.sortBy(_.map(_.keys(this._searchResult.genders), (x) => parseInt(x)));
      track.gender_id = keys[(_.indexOf(keys, track.gender_id) + 1) % keys.length];
    }

    // Merge track
    else if (chr == 'm') {
      this._pushState();
      this.props.group.elements[i-1].max_frame = this.props.group.elements[i].max_frame;
      this.props.group.elements.splice(i, 1);
    }

    // Delete track
    else if (chr == 'd') {
      this._pushState();
      this.props.group.elements.splice(i, 1);
    }

    // Change track topic
    else if (chr == 't') {
      this.setState({showSelect: true});
    }

    this.forceUpdate();
  }

  _onSelect = (value) => {
    let fps = this._video().fps;
    let curFrame = this.state.currentTime * fps;
    let curTrack = this.props.group.elements.map((clip, i) => [clip, i]).filter(([clip, _]) =>
      clip.min_frame <= curFrame && curFrame <= clip.max_frame)[0][1];
    this.props.group.elements[curTrack].thing_id = value;
    this.setState({showSelect: false});
  }

  _onKeyPress = (e) => {
    if (keyboardManager.locked() || !(this._videoPlaying || this.state.moused)) {
      return;
    }

    let chr = String.fromCharCode(e.which);

    let fps = this._video().fps;
    let curFrame = this.state.currentTime * fps;

    let elements = this.props.group.elements;
    if (chr == '\r') {
      let lastTrack = elements.map((clip, i) => [clip, i]).filter(([clip, _]) =>
        clip.min_frame <= curFrame);
      let offset = e.shiftKey ? -1 : 1;
      let index = lastTrack[lastTrack.length - 1][1] + offset;
      if (0 <= index && index < elements.length) {
        let newTime = elements[index].min_frame / fps + 0.1;
        this.setState({
          displayTime: newTime,
          currentTime: newTime
        });
      }
    }

    else if (chr == 'r') {
      let playbackSpeed = this._frontendSettings.get('playback_speed');
      if (playbackSpeed != 1) {
        this._frontendSettings.set('playback_speed', 1);
        this._lastPlaybackSpeed = playbackSpeed;
      } else {
        this._frontendSettings.set('playback_speed', this._lastPlaybackSpeed);
      }
    }

    else if (chr == 'i') {
      if (this.state.trackStart == -1) {
        this.setState({trackStart: this.state.currentTime});
      } else {
        this._pushState();

        let start = Math.round(this.state.trackStart * fps);
        let end = Math.round(this.state.currentTime * fps);

        let to_add = [];
        let to_delete = [];
        elements.map((clip, i) => {
          // +++ is the new clip, --- is old clip, overlap prioritized to new clip

          // [---[++]+++]
          if (clip.min_frame <= start && start <= clip.max_frame && clip.max_frame <= end) {
            clip.max_frame = start;
          }

          // [+++[+++]---]
          else if(start <= clip.min_frame && clip.min_frame <= end && end <= clip.max_frame){
            clip.min_frame = end;
          }

          // [---[+++]---]
          else if (clip.min_frame <= start && end <= clip.max_frame) {
            let new_clip = _.clone(clip);
            new_clip.min_frame = end;
            clip.max_frame = start;
            to_add.push(new_clip);
          }

          // [+++[+++]+++]
          else if (start <= clip.min_frame && clip.max_frame <= end) {
            to_delete.push(i);
          }
        });

        _.reverse(to_delete);
        to_delete.map((i) => elements.splice(i, -1));
        elements.push.apply(elements, to_add);
        elements.push({
          video: elements[0].video,
          min_frame: start,
          max_frame: end,
          gender_id: _.find(this._searchResult.genders, (l) => l.name == 'M').id
        });
        this.props.group.elements = _.sortBy(elements, ['min_frame']);

        this.setState({trackStart: -1});
      }
    }

    else if (chr == 'z') {
      if (this._undoStack.length > 0) {
        let lastState = this._undoStack.pop();
        this.props.group.elements = lastState;
      }
    }

    else {
      let curTracks = this.props.group.elements.map((clip, i) => [clip, i]).filter(([clip, _]) =>
        clip.min_frame <= curFrame && curFrame <= clip.max_frame);
      if (curTracks.length == 0) {
        console.warn('No tracks to process');
      } else if (curTracks.length > 1) {
        console.error('Attempting to process multiple tracks');
      } else {
        this._onTrackKeyPress(e, curTracks[0][1]);
      }
    }
  }

  _containerOnMouseOver = () => {
    this.setState({moused: true});
  }

  _containerOnMouseOut = () => {
    this.setState({moused: false});
  }

  componentDidMount() {
    document.addEventListener('keypress', this._onKeyPress);
    this.setState({
      currentTime: this.props.group.elements[0].min_frame / this._video().fps
    })
  }

  componentWillUnmount() {
    document.removeEventListener('keypress', this._onKeyPress);
  }

  // TODO(wcrichto): timeline disappears after deleting first track in the timeline

  render() {
    return <Consumer contexts={[FrontendSettingsContext, BackendSettingsContext, SearchContext]}>{(frontendSettings, backendSettings, searchResult) => {
        this._frontendSettings = frontendSettings;
        this._searchResult = searchResult;

        if (this._lastPlaybackSpeed === null) {
          this._lastPlaybackSpeed = this._frontendSettings.get('playback_speed');
        }

        let group = this.props.group;
        let expand = this.props.expand;

        let clip = {
          video: group.elements[0].video,
          min_frame: group.elements[0].min_frame,
          max_frame: group.elements[group.elements.length-1].max_frame
        };

        let video = this._video();
        let small_height = expand ? video.height : 100 * this._frontendSettings.get('thumbnail_size');
        let small_width = video.width * small_height / video.height;

        let style = {
          width: small_width,
        };

        let timeboxStyle = {
          width: small_width,
          height: expand ? 60 : 20
        }

        let tprops = {
          w: timeboxStyle.width,
          h: timeboxStyle.height,
          mw: expand ? 4 : 2,
          mh: timeboxStyle.height,
          mf: expand ? 16 : 12,
          currentTime: this.state.currentTime,
          video: video,
        };

        let selectWidth = 200;
        let selectStyle = {
          left: timeboxStyle.width / 2 - selectWidth / 2,
          top: small_height,
          position: 'absolute',
          zIndex: 1000
        };

        return <div className='timeline' style={style} onMouseOver={this._containerOnMouseOver} onMouseOut={this._containerOnMouseOut}>
          <ClipView clip={clip} onTimeUpdate={this._onTimeUpdate} showMeta={false}
                    expand={this.props.expand} displayTime={this.state.displayTime}
                    onVideoPlay={this._onVideoPlay}
                    onVideoStop={this._onVideoStop} />
          <svg className='time-container' style={timeboxStyle} onMouseDown={this._onMouseDown}
               onMouseMove={this._onMouseMove}
               onMouseUp={this._onMouseUp}
               onMouseOut={this._timelineOnMouseOut}
               ref={(n) => {this._svg = n;}}>
            <g>{group.elements.map((track, i) =>
              // We destructure the track b/c mobx doesn't seem to be observing updates to it?
              <TrackView key={i} i={i} track={track} onKeyPress={this._onTrackKeyPress} {...tprops} />)}
            </g>
            {this.state.trackStart != -1
             ? <MarkerView t={this.state.trackStart} type="open" color="rgb(230, 230, 20)" {...tprops} />
             : <g />}
            <line x1={tprops.w/2} x2={tprops.w/2} y1={0} y2={tprops.h} stroke="rgb(20, 230, 20)" strokeWidth={tprops.mw*1.5} />
            <MarkerView t={clip.min_frame/video.fps} type="open" color="black" {...tprops} />
            <MarkerView t={clip.max_frame/video.fps} type="close" color="black"  {...tprops} />
          </svg>
          {this.state.showSelect
           ? <div style={selectStyle}>
             <Select
               data={_.map(backendSettings.things['topic'], (v, k) => [k, v])}
               width={selectWidth}
               onSelect={this._onSelect}
               onClose={(e) => {this.setState({showSelect: false});}}
             />
           </div>
           : <div />}
        </div>;
    }}</Consumer>
  }
}
