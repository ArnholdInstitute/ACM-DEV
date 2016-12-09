import React from "react";
import * as BS from 'react-bootstrap';
import { LinkContainer } from 'react-router-bootstrap';
import * as _ from 'lodash'

export default class Nav extends React.Component {

  constructor(){
    super();
    this.state = {
      activeKey : 1
    };
  }
  render() {    
    return (
      <BS.Navbar style={{zIndex : 3}}>
        <BS.Navbar.Header>
          <BS.Navbar.Toggle/>
        </BS.Navbar.Header>
        <BS.Navbar.Collapse>
          <BS.Nav>
            <LinkContainer to="/landing">
              <BS.NavItem eventKey={1}>
                Home
              </BS.NavItem>
            </LinkContainer>
          </BS.Nav>
        </BS.Navbar.Collapse>
      </BS.Navbar>
    );
  }
}
