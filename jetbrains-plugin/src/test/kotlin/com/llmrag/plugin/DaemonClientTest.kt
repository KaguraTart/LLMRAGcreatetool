package com.llmrag.plugin

import kotlin.test.Test
import kotlin.test.assertTrue

class DaemonClientTest {
    @Test
    fun instantiateClient() {
        val client = DaemonClient("http://127.0.0.1:7474", 1)
        assertTrue(client is DaemonClient)
    }
}
