import React from 'react';
import axios from 'axios';
import {Link} from 'react-router-dom';
import {observer} from 'mobx-react';
import {observable} from 'mobx';
/*
 * export class VideoPlayer {
 *   @observable
 * };
 * */

@observer
export default class VideoSummary extends React.Component {
  constructor(props) {
    super(props);
    this.video = props.store;
    this.state = {show_video: false};
  }

  componentDidMount() {
    this._draw();
  }

  // TODO(wcrichto): bboxes can get off w/ video when skipping around a bunch?
  _draw() {
    if (this._video !== undefined) {
      let frame = Math.round(this._video.currentTime * this.video.fps);
      let ctx = this._canvas.getContext('2d');
      ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);

      if (frame in this.video.faces) {
        this.video.faces[frame].forEach((face) => {
          let x = face.bbox.x1;
          let y = face.bbox.y1;
          let w = face.bbox.x2 - x;
          let h = face.bbox.y2 - y;
          let scale = this._canvas.width / this.video.width;
          ctx.beginPath();
          ctx.lineWidth = '3';
          ctx.strokeStyle = 'red';
          ctx.rect(x * scale, y * scale, w * scale, h * scale);
          ctx.stroke();
        });
      }
    }
    requestAnimationFrame(this._draw.bind(this));
  }

  _onClickThumbnail() {
    this.setState({show_video: true});
    this.video.loadFaces();
  }

  render() {
    let video = this.video;
    let parts = video.path.split('/');
    let basename = parts[parts.length - 1];
    return (
      <div className='video-summary'>
        {this.props.show_meta
        ? <div><Link to={'/video/' + video.id}>{basename}</Link></div>
        : <div />}
        {!this.state.show_video
        ? (<img src={"/static/thumbnails/" + video.id + ".jpg"}
                onClick={this._onClickThumbnail.bind(this)} />)
         : (this.video.loadedFaces == 0
          ? (<div>Loading...</div>)
          : (<div>
            <canvas ref={(n) => { this._canvas = n; }}></canvas>
            <video controls ref={(n) => { this._video = n; }}>
              <source src={"/fs/usr/src/app/" + video.path} />
            </video>
          </div>))}
      </div>
    );
  }
};
