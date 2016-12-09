import React, {Component} from 'react';
import Nav from './components/Nav'

export default class Layout extends React.Component {
  render() {
    return (
      <div style={{height : '100%', width : '100%'}}>
      	<Nav/>
        	{this.props.children}
      </div>
    );
  }
}

