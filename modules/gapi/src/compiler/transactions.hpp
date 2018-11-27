// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_COMPILER_TRANSACTIONS_HPP
#define OPENCV_GAPI_COMPILER_TRANSACTIONS_HPP

#include <algorithm> // find_if
#include <functional>
#include <list>

#include <ade/graph.hpp>

#include "opencv2/gapi/own/assert.hpp"

enum class Direction: int {Invalid, In, Out};

////////////////////////////////////////////////////////////////////////////
////
// TODO: Probably it can be moved to ADE

namespace Change
{
    struct Base
    {
        virtual void commit  (ade::Graph & ) {};
        virtual void rollback(ade::Graph & ) {};
        virtual ~Base() = default;
    };

    class NodeCreated final: public Base
    {
        ade::NodeHandle m_node;
    public:
        explicit NodeCreated(const ade::NodeHandle &nh) : m_node(nh) {}
        virtual void rollback(ade::Graph &g) override { g.erase(m_node); }
    };

    // NB: Drops all metadata stored in the EdgeHandle,
    // which is not restored even in the rollback

    // FIXME: either add a way for users to preserve meta manually
    // or extend ADE to manipulate with meta such way
    class DropLink final: public Base
    {
        ade::NodeHandle m_node;
        Direction       m_dir;

        ade::NodeHandle m_sibling;

    public:
        DropLink(ade::Graph &g,
                 const ade::NodeHandle &node,
                 const ade::EdgeHandle &edge)
            : m_node(node), m_dir(node == edge->srcNode()
                                  ? Direction::Out
                                  : Direction::In)
        {
            m_sibling = (m_dir == Direction::In
                         ? edge->srcNode()
                         : edge->dstNode());
            g.erase(edge);
        }

        virtual void rollback(ade::Graph &g) override
        {
            switch(m_dir)
            {
            case Direction::In:  g.link(m_sibling, m_node); break;
            case Direction::Out: g.link(m_node, m_sibling); break;
            default: GAPI_Assert(false);
            }
        }
    };

    class NewLink final: public Base
    {
        ade::EdgeHandle m_edge;

    public:
        NewLink(ade::Graph &g,
                  const ade::NodeHandle &prod,
                  const ade::NodeHandle &cons)
            : m_edge(g.link(prod, cons))
        {
        }

        virtual void rollback(ade::Graph &g) override
        {
            g.erase(m_edge);
        }
    };

    class DropNode final: public Base
    {
        ade::NodeHandle m_node;

    public:
        explicit DropNode(const ade::NodeHandle &nh)
            : m_node(nh)
        {
            // According to the semantic, node should be disconnected
            // manually before it is dropped
            GAPI_Assert(m_node->inEdges().size()  == 0);
            GAPI_Assert(m_node->outEdges().size() == 0);
        }

        virtual void commit(ade::Graph &g) override
        {
            g.erase(m_node);
        }
    };

    class List
    {
        std::list< std::unique_ptr<Base> > m_changes;

    public:
        template<typename T, typename ...Args>
        void enqueue(Args&&... args)
        {
            std::unique_ptr<Base> p(new T(args...));
            m_changes.push_back(std::move(p));
        }

        void commit(ade::Graph &g)
        {
            // Commit changes in the forward order
            for (auto& ch : m_changes) ch->commit(g);
        }

        void rollback(ade::Graph &g)
        {
            // Rollback changes in the reverse order
            for (auto it = m_changes.rbegin(); it != m_changes.rend(); ++it)
            {
                (*it)->rollback(g);
            }
        }
    };
} // namespace Change
////////////////////////////////////////////////////////////////////////////

#endif // OPENCV_GAPI_COMPILER_TRANSACTIONS_HPP
