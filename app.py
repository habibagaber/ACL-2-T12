import streamlit as st
import json
import plotly.graph_objects as go
import networkx as nx
from graph_rag import GraphRAGSystem
from llm import LLMLayer
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Graph-RAG Travel Intelligence Platform",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-weight: 500;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    try:
        config = {}
        with open('config.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    config[k.strip()] = v.strip()
        
        st.session_state.rag_system = GraphRAGSystem(
            uri=config.get('URI', 'neo4j://localhost:7687'),
            user=config.get('USERNAME', 'neo4j'),
            password=config.get('PASSWORD', 'neo4j')
        )
        st.session_state.llm_layer = LLMLayer()
        st.session_state.embeddings_created = False
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        st.stop()

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ System Configuration")
    
    # Model selection
    st.markdown("### Language Model")
    model_options = {
        'Zephyr 7B (Recommended)': 'zephyr-7b',
        'Mistral 7B': 'mistral-7b',
        'Llama 3.2 3B': 'llama-3.2-3b',
        'Neural Chat': 'neural-chat',
        'Gemma 2B': 'gemma-2b',
    }
    selected_model = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0,
        help="Choose the language model for generating responses"
    )
    model_name = model_options[selected_model]
    
    if 'zephyr' in model_name or 'llama-3.2' in model_name:
        st.success("✓ High-performance model selected")
    
    st.markdown("---")
    
    # Retrieval configuration
    st.markdown("### Retrieval Strategy")
    retrieval_method = st.radio(
        "Method",
        ["Baseline (Cypher)", "Embeddings", "Hybrid (Both)"],
        index=2,
        help="Select the knowledge retrieval approach"
    )
    
    # Add explanation of what each method does
    if retrieval_method == "Baseline (Cypher)":
        st.info("🔍 **Cypher Queries Only**: Uses predefined graph patterns for exact matching")
    elif retrieval_method == "Embeddings":
        st.info("🧠 **Semantic Search Only**: Uses AI embeddings for meaning-based retrieval")
    else:
        st.info("⚡ **Hybrid Mode**: Combines both methods for comprehensive results")
    
    # Embedding configuration
    if retrieval_method in ["Embeddings", "Hybrid (Both)"]:
        st.markdown("#### Embedding Configuration")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            index=0,
            help="Choose the model for semantic embeddings"
        )
        st.session_state.rag_system.current_embedding_model = embedding_model
        
        if not st.session_state.embeddings_created:
            if st.button("Initialize Embeddings", type="secondary"):
                with st.spinner("Generating node embeddings..."):
                    st.session_state.rag_system.create_node_embeddings(embedding_model)
                    st.session_state.embeddings_created = True
                    st.success("Embeddings initialized successfully")
    
    st.markdown("---")
    
    # Model comparison
    st.markdown("### Advanced Options")
    comparison_mode = st.checkbox("Enable Multi-Model Comparison")
    if comparison_mode:
        comparison_models = st.multiselect(
            "Models to Compare",
            list(model_options.values()),
            default=['zephyr-7b', 'mistral-7b', 'llama-3.2-3b'],
            help="Select 2 or more models for side-by-side comparison"
        )
    
    st.markdown("---")
    
    # System information
    st.markdown("### System Information")
    st.caption("**Platform:** Graph-RAG Intelligence")
    st.caption("**Version:** 1.0.0")
    st.caption("**Backend:** Neo4j Knowledge Graph")

# Main header
st.markdown('<h1 class="main-header">🌐 Graph-RAG Travel Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise-grade travel recommendations powered by knowledge graphs and advanced language models</p>', unsafe_allow_html=True)

# Create tabs with professional naming
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Query Interface", 
    "📊 Model Analytics", 
    "🗺️ Graph Explorer",
    "📈 System Insights"
])

# Tab 1: Query Interface
with tab1:
    st.markdown("### Intelligent Query System")
    st.caption("Ask natural language questions about hotels, destinations, and travel requirements")
    
    # Sample queries in a more professional format
    with st.expander("💡 Example Queries", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Hotel Search**")
            st.markdown("""
            • Hotels in Paris with ratings above 8.0
            • Top 5 hotels in Cairo
            • 4-star hotels in Madrid
            • Value for money hotels in Berlin
            """)
            st.markdown("**Travel Information**")
            st.markdown("""
            • Visa requirements from Egypt to France
            • Hotel statistics in Amsterdam
            • Average ratings by city
            """)
        with col2:
            st.markdown("**Reviews & Feedback**")
            st.markdown("""
            • Customer reviews for hotels in Dubai
            • What do travelers say about Barcelona
            • Hotel facilities in London
            """)
            st.markdown("**Personalized Recommendations**")
            st.markdown("""
            • Hotels for solo travelers in Barcelona
            • Family-friendly accommodations in Rome
            • Hotels suitable for seniors in Cairo
            """)
    
    # Query input with professional styling
    user_query = st.text_input(
        "Enter Query",
        placeholder="e.g., Find top-rated hotels in Paris with excellent facilities",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        query_button = st.button("Execute Query", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()
    
    if query_button and user_query:
        with st.spinner("Processing query and retrieving knowledge..."):
            # Map UI choice to method parameter
            method_map = {
                "Baseline (Cypher)": "baseline",
                "Embeddings": "embeddings",
                "Hybrid (Both)": "hybrid"
            }
            selected_method = method_map[retrieval_method]
            
            # Retrieve from knowledge graph with selected method
            context = st.session_state.rag_system.combined_retrieval(
                user_query, 
                retrieval_method=selected_method
            )
            
            # Display query analysis
            st.markdown("#### Query Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                intents = ', '.join(context['intents'])
                st.info(f"**Detected Intent:** {intents}")
            with col2:
                entities_str = ", ".join([f"{k}: {v}" for k, v in context['entities'].items() if v])
                st.info(f"**Extracted Entities:** {entities_str if entities_str else 'None'}")
            with col3:
                # Show which method was actually used
                method_display = context.get('retrieval_method', 'hybrid').title()
                st.info(f"**Method Used:** {method_display}")
            
            # Show fallback warning if applicable
            if context['baseline'].get('fallback_used'):
                st.warning("⚠️ Fallback to baseline retrieval was triggered (embeddings unavailable or unsuitable for query type)")
            
            # Retrieved context (collapsible)
            with st.expander("📦 Knowledge Graph Context", expanded=False):
                st.markdown("##### Cypher Query Results")
                if context['baseline']['data']:
                    st.json(context['baseline']['data'][:5])
                else:
                    st.warning("No results retrieved from baseline queries")
                
                if context['embedding']:
                    st.markdown("##### Semantic Search Results")
                    st.json(context['embedding']['data'])
            
            # Executed queries
            with st.expander("💾 Database Queries Executed", expanded=False):
                for i, query_info in enumerate(context['baseline']['cypher_queries'], 1):
                    st.markdown(f"**Query {i}**")
                    st.code(query_info['query'], language='cypher')
                    st.json(query_info['params'])
            
            # AI Response
            st.markdown("#### Generated Response")
            
            if not comparison_mode:
                # Single model response
                llm_layer = st.session_state.llm_layer
                prompt = llm_layer.create_prompt(context, user_query)
                
                with st.spinner(f"Generating response using {selected_model}..."):
                    result = llm_layer.query_model(prompt, model_name, max_tokens=500)
                
                if result['success']:
                    st.markdown(f"**{selected_model}**")
                    st.success(result['response'])
                    
                    # Performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{result['time_seconds']:.2f}s")
                    with col2:
                        st.metric("Tokens Generated", result.get('tokens_used', 'N/A'))
                    with col3:
                        st.metric("API Cost", f"${result.get('cost', 0):.4f}")
                else:
                    st.warning(f"Model API unavailable. Displaying raw data:")
                    st.info(result.get('response', 'No response available'))
                    if result.get('error'):
                        with st.expander("Error Details"):
                            st.error(result['error'])
            
            else:
                # Multi-model comparison
                st.markdown("### Multi-Model Comparison")
                
                comparison_results = st.session_state.llm_layer.compare_models(
                    context, user_query, comparison_models
                )
                
                for model, result in comparison_results['results'].items():
                    with st.expander(f"**{model.upper()}**", expanded=True):
                        if result['success']:
                            st.markdown(result['response'])
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Time", f"{result['time_seconds']:.2f}s")
                            with col2:
                                st.metric("Tokens", result.get('tokens_used', 'N/A'))
                            with col3:
                                st.metric("Cost", f"${result.get('cost', 0):.4f}")
                        else:
                            st.error(f"Model error: {result.get('error', 'Unknown error')}")

# Tab 2: Model Analytics
with tab2:
    st.markdown("### Model Performance Analytics")
    st.caption("Comprehensive evaluation and comparison of language model performance")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        test_query = st.text_input(
            "Benchmark Query",
            value="What are the best hotels in Paris?",
            key="comparison_query",
            help="Enter a query to benchmark across multiple models"
        )
    
    with col2:
        st.markdown("###")  # Spacing
        models_to_compare = st.multiselect(
            "Select Models",
            list(model_options.values()),
            default=['zephyr-7b', 'mistral-7b', 'llama-3.2-3b'],
            help="Choose 2 or more models for comparison"
        )
    
    if st.button("🚀 Execute Benchmark", type="primary") and len(models_to_compare) >= 2:
        with st.spinner("Running comprehensive model evaluation..."):
            # Map UI choice to method parameter
            method_map = {
                "Baseline (Cypher)": "baseline",
                "Embeddings": "embeddings",
                "Hybrid (Both)": "hybrid"
            }
            selected_method = method_map[retrieval_method]
            
            # Get context with selected method
            context = st.session_state.rag_system.combined_retrieval(
                test_query,
                retrieval_method=selected_method
            )
            
            # Compare models
            comparison = st.session_state.llm_layer.compare_models(
                context, test_query, models_to_compare
            )
            
            # Quantitative metrics
            st.markdown("#### Quantitative Metrics")
            
            metrics_data = []
            for model in models_to_compare:
                result = comparison['results'][model]
                if result['success']:
                    metrics_data.append({
                        'Model': model.upper(),
                        'Response Time (s)': round(result.get('time_seconds', 0), 3),
                        'Tokens Generated': result.get('tokens_used', 0),
                        'API Cost ($)': round(result.get('cost', 0), 6),
                        'Response Length': len(result.get('response', ''))
                    })
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Performance visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_time = go.Figure(data=[
                        go.Bar(
                            x=df['Model'], 
                            y=df['Response Time (s)'],
                            marker_color='rgb(102, 126, 234)',
                            text=df['Response Time (s)'],
                            textposition='auto',
                        )
                    ])
                    fig_time.update_layout(
                        title="Response Time Analysis",
                        xaxis_title="Model",
                        yaxis_title="Time (seconds)",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    fig_cost = go.Figure(data=[
                        go.Bar(
                            x=df['Model'], 
                            y=df['API Cost ($)'],
                            marker_color='rgb(118, 75, 162)',
                            text=df['API Cost ($)'],
                            textposition='auto',
                        )
                    ])
                    fig_cost.update_layout(
                        title="Cost Analysis",
                        xaxis_title="Model",
                        yaxis_title="Cost (USD)",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                # Performance summary
                summary = comparison['summary']
                st.markdown("#### Performance Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fastest Model", summary['fastest_model'].upper(), delta="Optimal")
                with col2:
                    st.metric("Most Cost-Effective", summary['cheapest_model'].upper(), delta="Optimal")
                with col3:
                    st.metric("Most Verbose", summary['longest_response'].upper(), delta="Highest")
            
            # Qualitative comparison
            st.markdown("#### Qualitative Response Comparison")
            for model in models_to_compare:
                result = comparison['results'][model]
                with st.expander(f"**{model.upper()}** Response Analysis", expanded=False):
                    if result['success']:
                        st.markdown(result['response'])
                        st.caption(f"Generated in {result['time_seconds']:.2f}s | {result.get('tokens_used', 'N/A')} tokens")
                    else:
                        st.error(f"Model unavailable: {result.get('error', 'Failed to generate response')}")

# Tab 3: Graph Explorer
with tab3:
    st.markdown("### Knowledge Graph Visualization")
    st.caption("Interactive exploration of the underlying knowledge graph structure")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        viz_query = st.text_input(
            "Visualization Query",
            value="Hotels in Paris",
            key="viz_query",
            help="Enter a query to visualize the related knowledge subgraph"
        )
    with col2:
        st.markdown("###")  # Spacing
        if st.button("Generate Visualization", type="primary", use_container_width=True):
            with st.spinner("Constructing graph visualization..."):
                # Map UI choice to method parameter
                method_map = {
                    "Baseline (Cypher)": "baseline",
                    "Embeddings": "embeddings",
                    "Hybrid (Both)": "hybrid"
                }
                selected_method = method_map[retrieval_method]
                
                context = st.session_state.rag_system.combined_retrieval(
                    viz_query,
                    retrieval_method=selected_method
                )
                
                # Create NetworkX graph
                G = nx.Graph()
                
                # Build graph from results
                hotels = []
                for item in context['combined_data'][:10]:
                    if 'name' in item:
                        hotel_name = item['name']
                        hotels.append(hotel_name)
                        G.add_node(hotel_name, 
                                  node_type='hotel',
                                  rating=item.get('avg_score', 0))
                        
                        if 'city' in item:
                            city = item['city']
                            G.add_node(city, node_type='city')
                            G.add_edge(hotel_name, city)
                        
                        if 'country' in item:
                            country = item['country']
                            G.add_node(country, node_type='country')
                            if 'city' in item:
                                G.add_edge(item['city'], country)
                
                if len(G.nodes()) > 0:
                    # Create visualization
                    pos = nx.spring_layout(G, k=0.5, iterations=50)
                    
                    # Edge traces
                    edge_trace = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace.append(
                            go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=2, color='rgba(156, 163, 175, 0.5)'),
                                hoverinfo='none',
                                showlegend=False
                            )
                        )
                    
                    # Node trace
                    node_trace = go.Scatter(
                        x=[],
                        y=[],
                        text=[],
                        mode='markers+text',
                        hoverinfo='text',
                        marker=dict(
                            size=[],
                            color=[],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title="Rating",
                                thickness=15,
                                len=0.7
                            ),
                            line=dict(width=2, color='white')
                        ),
                        textposition="top center",
                        textfont=dict(size=10, color='#1f2937')
                    )
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_trace['x'] += tuple([x])
                        node_trace['y'] += tuple([y])
                        node_trace['text'] += tuple([node])
                        
                        node_type = G.nodes[node].get('node_type', 'other')
                        if node_type == 'hotel':
                            node_trace['marker']['size'] += tuple([25])
                            node_trace['marker']['color'] += tuple([G.nodes[node].get('rating', 5)])
                        elif node_type == 'city':
                            node_trace['marker']['size'] += tuple([35])
                            node_trace['marker']['color'] += tuple([8])
                        else:
                            node_trace['marker']['size'] += tuple([45])
                            node_trace['marker']['color'] += tuple([10])
                    
                    # Create figure
                    fig = go.Figure(data=edge_trace + [node_trace],
                                  layout=go.Layout(
                                      title=dict(
                                          text=f'Knowledge Graph: {viz_query}',
                                          font=dict(size=20, color='#1f2937')
                                      ),
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20, l=20, r=20, t=60),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      height=600,
                                      plot_bgcolor='#f9fafb'
                                  ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Graph statistics
                    st.markdown("#### Graph Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Nodes", len(G.nodes()))
                    with col2:
                        st.metric("Total Edges", len(G.edges()))
                    with col3:
                        st.metric("Hotels Displayed", len(hotels))
                    with col4:
                        density = nx.density(G)
                        st.metric("Graph Density", f"{density:.3f}")
                else:
                    st.warning("No graph data available for the specified query")

# Tab 4: System Insights
with tab4:
    st.markdown("### System Architecture & Performance")
    st.caption("Technical overview and performance characteristics of the Graph-RAG platform")
    
    # Architecture overview
    st.markdown("#### System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Input Processing Pipeline")
        st.markdown("""
        **Intent Classification**  
        • Rule-based pattern matching  
        • Multi-intent detection  
        • Context-aware entity extraction
        
        **Entity Extraction**  
        • Named Entity Recognition (NER)  
        • Fuzzy matching for locations  
        • Numeric parameter parsing
        
        **Query Embedding**  
        • Semantic vector representation  
        • Multiple embedding models  
        • Similarity-based retrieval
        """)
        
    with col2:
        st.markdown("##### Knowledge Retrieval Layer")
        st.markdown("""
        **Baseline Retrieval**  
        • 14 predefined Cypher templates  
        • Direct graph pattern matching  
        • Structured query execution
        
        **Semantic Retrieval**  
        • Node embeddings (2 models)  
        • Cosine similarity ranking  
        • Top-k selection
        
        **Hybrid Strategy**  
        • Combined approach  
        • Result deduplication  
        • Intelligent fallback
        """)
    
    st.markdown("---")
    
    # Model information
    st.markdown("#### Available Language Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Free Tier Models")
        models_info = pd.DataFrame({
            'Model': ['Zephyr 7B', 'Mistral 7B', 'Llama 3.2 3B', 'Neural Chat', 'Gemma 2B'],
            'Parameters': ['7B', '7B', '3B', '7B', '2B'],
            'Status': ['✓ Active', '✓ Active', '✓ Active', '✓ Active', '✓ Active'],
            'Quality': ['High', 'High', 'Medium', 'Medium', 'Medium']
        })
        st.dataframe(models_info, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown("##### Integration Options")
        st.markdown("""
        **Available APIs**  
        • HuggingFace Inference API  
        • Serverless Inference Endpoints  
        • Dedicated Inference Endpoints
        
        **Authentication**  
        • Environment variable token  
        • Secure API key management  
        • Rate limiting compliance
        
        **Extensibility**  
        • OpenAI integration ready  
        • Anthropic Claude support  
        • Custom model deployment
        """)
    
    st.markdown("---")
    
    # Performance characteristics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("Known Limitations"):
            st.markdown("""
            • Entity extraction accuracy depends on input quality
            • Embedding search requires preprocessing
            • LLM responses subject to model availability
            • Cypher queries limited to predefined templates
            • API rate limiting on free tier
            • Potential model hallucination
            """)
    
    with col2:
        with st.expander("Implemented Optimizations"):
            st.markdown("""
            • Hybrid retrieval strategy
            • Intelligent result deduplication
            • Structured prompt engineering
            • Multi-model embedding support
            • Triple-fallback error handling
            • Smart response generation
            • Batch embedding processing
            """)
    
    with col3:
        with st.expander("Roadmap & Enhancements"):
            st.markdown("""
            • Advanced NER integration
            • Query expansion techniques
            • User feedback mechanisms
            • Dynamic Cypher generation
            • Response caching layer
            • Enhanced visualizations
            • Multi-language support
            • Model health monitoring
            """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("**Graph-RAG Travel Intelligence Platform** | Version 1.0.0")
with col2:
    st.caption("Powered by Neo4j & Advanced NLP")
with col3:
    st.caption("CSEN 903 Project")

# Help section in sidebar
with st.sidebar:
    st.markdown("---")
    if st.checkbox("Setup Instructions"):
        st.markdown("#### HuggingFace Configuration")
        st.code("""
# Obtain token from:
# https://huggingface.co/settings/tokens

# Windows Command Prompt:
set HUGGINGFACE_TOKEN=hf_xxx

# Windows PowerShell:
$env:HUGGINGFACE_TOKEN="hf_xxx"

# Linux/macOS:
export HUGGINGFACE_TOKEN=hf_xxx
""", language="bash")
        st.caption("Token required for API access")