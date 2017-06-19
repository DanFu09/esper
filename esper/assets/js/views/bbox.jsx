import React from 'react';
import {observer} from 'mobx-react';
import {observable, autorun} from 'mobx';

export let boundingRect = (div) => {
  let r = div.getBoundingClientRect();
  return {
    left: r.left + document.body.scrollLeft,
    top: r.top + document.body.scrollTop,
    width: r.width,
    height: r.height
  };
};

export class Box {
  @observable x
  @observable y
  @observable w
  @observable h
  @observable cls
  @observable track

  constructor(x1, y1, x2, y2, cls, track) {
    this.x = x1;
    this.y = y1;
    this.w = x2 - x1;
    this.h = y2 - y1;
    this.cls = cls;
    this.track = track;
  }

  rescale(scale) {
    this.x *= scale;
    this.y *= scale;
    this.w *= scale;
    this.h *= scale;
  }
}

@observer
class BoxView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      box: this.props.box,
      clickX: -1,
      clickY: -1,
      clicked: false
    };
    this._mouseX = -1;
    this._mouseY = -1;
  }

  _onMouseDown(e) {
    let rect = boundingRect(this._div);
    this.setState({
      clicked: true,
      clickX: e.pageX - rect.left,
      clickY: e.pageY - rect.top
    });
    e.stopPropagation();
  }

  _onMouseMove(e) {
    let rect = boundingRect(this._div);
    let ox = rect.left - this.state.box.x;
    let oy = rect.top - this.state.box.y;
    this._mouseX = e.pageX - ox;
    this._mouseY = e.pageY - oy;
    if (!this.state.clicked) { return; }
    let box = this.state.box;
    box.x = this._mouseX - this.state.clickX;
    box.y = this._mouseY - this.state.clickY;
  }

  _onMouseUp(e) {
    this.setState({clicked: false});
    e.stopPropagation();
  }

  _onKeyDown(e) {
    let chr = String.fromCharCode(e.which);
    let box = this.state.box;
    let covers =
      box.x <= this._mouseX && this._mouseX <= box.x + box.w &&
      box.y <= this._mouseY && this._mouseY <= box.y + box.h;
    if (chr == ' ') {
      if (covers) {
        let cls = 'gender-F';
        if (box.cls == 'gender-F') {
          cls = 'gender-M';
        }
        box.cls = cls;
        this.props.onChange();
      }

      e.preventDefault();
    } else if (chr == 'D' && covers) {
      this.props.onDelete();
    } else if(chr == 'T' && covers) {
      this.props.onTrack();
    }
  }

  componentDidMount() {
    this._div.addEventListener('mouseup', this._onMouseUp.bind(this));
    this._mouseMoveListener = this._onMouseMove.bind(this);
    document.addEventListener('mousemove', this._mouseMoveListener);
    this._div.addEventListener('mousedown', this._onMouseDown.bind(this));
    this._keyDownListener = this._onKeyDown.bind(this);
    document.addEventListener('keydown', this._keyDownListener);
  }

  componentWillUnmount() {
    document.removeEventListener('mousemove', this._mouseMoveListener);
    document.removeEventListener('keydown', this._keyDownListener);
  }

  render() {
    let box = this.state.box;
    let style = {
      left: box.x,
      top: box.y,
      width: box.w,
      height: box.h
    };

    return <div className={`bounding-box ${box.cls}`}
                style={style}
                ref={(n) => {this._div = n}}>{box.track}</div>;
  }
}

export class BoundingBoxView extends React.Component {
  state = {
    startX: -1,
    startY: -1,
    curX: -1,
    curY: -1,
    mouseIn: false,
    fullwidth: false,
  }

  componentWillReceiveProps(props) {
    let {width, height} = this._getDimensions();
    let img_height = height;
    let scale = img_height / this.props.height;
    props.bboxes.forEach((bbox) => {
      bbox.rescale(scale);
    });
  }

  _onMouseDown(e) {
    let rect = boundingRect(this._div);
    this._ox = rect.left;
    this._oy = rect.top;
    this.setState({
      startX: e.pageX - this._ox,
      startY: e.pageY - this._oy
    });
  }

  _onMouseMove(e) {
    if (!this._div) { return; }
    let rect = boundingRect(this._div);
    this._ox = rect.left;
    this._oy = rect.top;
    this._ow = rect.width;
    this._oh = rect.height;
    let curX = e.pageX - this._ox;
    let curY = e.pageY - this._oy;
    if (0 <= curX && curX <= this._ow &&
        0 <= curY && curY <= this._oh) {
      this.setState({curX: curX, curY: curY});
    }
  }

  _onMouseUp(e) {
    this.props.bboxes.push(new Box(
      this.state.startX, this.state.startY,
      this.state.curX, this.state.curY,
      'gender-0', null));
    this.setState({startX: -1});
  }

  _onKeyDown(e) {
    let chr = String.fromCharCode(e.which);
    let {width, height} = this._getDimensions();
    if (chr == 'F' && this.state.mouseIn) {
      if (this.state.fullwidth) {
        let scale = 1 / this._lastScale;
        this.props.bboxes.forEach((box) => {
          box.rescale(scale);
        });
        this.setState({fullwidth: false});
      } else {
        let containerWidth = 780;
        let scale = containerWidth / width;
        this._lastScale = scale;
        this.props.bboxes.forEach((box) => {
          box.rescale(scale);
        });
        this.setState({fullwidth: true});
      }
    }
  }

  _getDimensions() {
    let rect = boundingRect(this._div);
    return {width: rect.width, height: rect.height};
  }

  componentDidMount() {
    this._div.addEventListener('mousedown', this._onMouseDown.bind(this));
    document.addEventListener('mousemove', this._onMouseMove.bind(this));
    this._div.addEventListener('mouseup', this._onMouseUp.bind(this));
    document.addEventListener('keydown', this._onKeyDown.bind(this));
    this._div.addEventListener('mouseover', (() => {
      this.setState({mouseIn: true});
    }).bind(this));
    this._div.addEventListener('mouseout', (() => {
      this.setState({mouseIn: false});
    }).bind(this));
  }

  _onDelete(i) {
    this.props.bboxes.splice(i, 1);
  }

  _onChange(i) {
    let box = this.props.bboxes[i];
    this.props.onChange(box);
  }

  _onTrack(i) {
    let box = this.props.bboxes[i];
    this.props.onTrack(box);
  }

  render() {
    let imgStyle = this.state.fullwidth ? {width: '780px', height: 'auto'} : {};
    return (
      <div ref={(n) => {this._div = n;}} className='bounding-boxes'>
        {this.state.startX != -1
         ? <BoxView box={new Box(this.state.startX,
                                 this.state.startY,
                                 this.state.curX,
                                 this.state.curY,
                                 'gender-0', null)} />
         : <div />}
        {this.props.bboxes.map((box, i) =>
          <BoxView fullwidth={this.state.fullwidth} box={box} key={i}
                   onDelete={() => this._onDelete(i)}
                   onChange={() => this._onChange(i)}
                   onTrack={() => this._onTrack(i)} />)}
        <img ref={(n) => {this._img = n;}} src={this.props.path} draggable={false}
             style={imgStyle} />
      </div>
    );
  }
};
