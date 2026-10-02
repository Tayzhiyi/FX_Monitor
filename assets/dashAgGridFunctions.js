var dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};

// Sort parent and child rows by their pair key. postSortRows then restores each
// expanded pair's child rows immediately after the pair summary.
dagfuncs.comparePairRows = function (valueA, valueB, nodeA, nodeB) {
    const pairA = (nodeA && nodeA.data && nodeA.data.pair_key) || "";
    const pairB = (nodeB && nodeB.data && nodeB.data.pair_key) || "";
    return pairA.localeCompare(pairB);
};

dagfuncs.keepTradesWithPair = function (params) {
    const parents = [];
    const childrenByPair = Object.create(null);

    params.nodes.forEach(function (node) {
        const data = node.data || {};
        const pair = data.pair_key || "";

        if (data.row_type === "trade") {
            if (!childrenByPair[pair]) {
                childrenByPair[pair] = [];
            }
            childrenByPair[pair].push(node);
        } else {
            parents.push(node);
        }
    });

    Object.keys(childrenByPair).forEach(function (pair) {
        childrenByPair[pair].sort(function (a, b) {
            const dateA = (a.data && a.data.trade_date) || "";
            const dateB = (b.data && b.data.trade_date) || "";
            if (dateA !== dateB) {
                return dateA.localeCompare(dateB);
            }

            const idA = (a.data && a.data.trade_id) || "";
            const idB = (b.data && b.data.trade_id) || "";
            return idA.localeCompare(idB);
        });
    });

    const sortedRows = [];
    parents.forEach(function (parent) {
        sortedRows.push(parent);
        const pair = (parent.data && parent.data.pair_key) || "";
        (childrenByPair[pair] || []).forEach(function (child) {
            sortedRows.push(child);
        });
    });

    params.nodes.length = 0;
    sortedRows.forEach(function (node) {
        params.nodes.push(node);
    });
};
