package com.llmrag.plugin

import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.components.JBTextArea
import com.intellij.ui.content.ContentFactory
import java.awt.BorderLayout
import javax.swing.JButton
import javax.swing.JPanel
import javax.swing.JTextField

class LLMRAGToolWindowFactory : ToolWindowFactory, DumbAware {
    override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
        val daemon = DaemonClient()
        val panel = JPanel(BorderLayout())

        val queryField = JTextField("What is in my workspace?")
        val outputArea = JBTextArea()
        outputArea.isEditable = false
        val runButton = JButton("Query")

        runButton.addActionListener {
            val workspace = "default"
            val response = runCatching { daemon.query(workspace, queryField.text, 5) }
                .getOrElse { "Error: ${it.message}" }
            outputArea.text = response
        }

        panel.add(queryField, BorderLayout.NORTH)
        panel.add(JBScrollPane(outputArea), BorderLayout.CENTER)
        panel.add(runButton, BorderLayout.SOUTH)

        val content = ContentFactory.getInstance().createContent(panel, "LLM RAG", false)
        toolWindow.contentManager.addContent(content)
    }

    override fun shouldBeAvailable(project: Project): Boolean = true
}
